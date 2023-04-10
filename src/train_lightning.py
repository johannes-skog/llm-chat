import os
import torch
import sys
from datasets import load_from_disk
from azureml.core.model import Model
from model import GeneratorModel
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.constants import AssetTypes
from util import (
    get_ml_client,
    download_dataset,
    download_model,
    DataNames,
    create_traced_model
)
import torchmetrics
from lightning.pytorch.cli import LightningCLI
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import transformers
from util import AzureBlobStorageCheckpoint
from model import Gpt2, Llama
import argparse
import yaml
import json


class DataModule(pl.LightningDataModule):
    
    DATASET_PATH = f"artifacts/dataset"
    
    def __init__(
        self,
        dataset_name: str,
        batch_size_train: int = 1,
        workers: int = 2,
        subset_train: int = None,
        local: bool = False,
    ):
        
        super().__init__()
        
        if local is False:

            download_dataset(
                ml_client=get_ml_client(),
                name=dataset_name,
                destination=DataModule.DATASET_PATH,
            )
        
        self.local = local
        self.dataset_name = dataset_name
        self.workers = workers
        self.subset_train = subset_train
        self.batch_size_train = batch_size_train

    def setup(self, stage: str):
        
        self._dataset_hg = load_from_disk(DataModule.DATASET_PATH)
        self._dataset_hg = self._dataset_hg["train"].train_test_split(test_size=0.01)

    def train_dataloader(self):
        
        dataset = self._dataset_hg["train"]

        if self.subset_train is not None:
            dataset = dataset.select(range(self.subset_train))
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size_train,
            shuffle=False,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        
        dataset = self._dataset_hg["test"].select(range(8))

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size_train,
            shuffle=False,
            num_workers=self.workers,
        )

    def test_dataloader(self):
        None

    def predict_dataloader(self):
        None

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...

class ModLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--run_name", default=None)

class CustomCallback(pl.callbacks.Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")

def setup_callbacks():

    callbacks = []

    checkpoint_callback_spaced = ModelCheckpoint(
        every_n_train_steps=1000,
    )
    # Some bug in lightning, this needs to be set manually outside of
    # the constructor..complains aboout monitor not set otherwise
    # checkpoint_callback_spaced.save_top_k = 2

    callbacks.append(checkpoint_callback_spaced)

    # Save to Azure blob storage
    if False:

        checkpoint_az = AzureBlobStorageCheckpoint(
            connection_string=os.getenv("AZURE_CHECKPOINT_CONNECTION_STRING"),
            container_name=os.getenv("AZURE_CHECKPOINT_CONTAINER_NAME"),
            save_top_k=2,
            monitor="loss_val",
            mode="min",
            filename='{epoch}-{loss_val:.4f}',
        )
        callbacks.append(checkpoint_az)

    else:

        # Save the two best checkpoints
        checkpoint_callback_val = ModelCheckpoint(
            save_top_k=2,
            monitor="loss_val",
            mode="min",
            filename='{epoch}-{loss_val:.4f}',
        )

        callbacks.append(checkpoint_callback_val)

    return callbacks, checkpoint_callback_spaced

def cli_main():

    callbacks, checkpointer = setup_callbacks()
    
    cli = ModLightningCLI(
        save_config_overwrite=True,
        model_class=GeneratorModel,
        trainer_class=pl.Trainer,
        datamodule_class=DataModule,
        run=False,
        trainer_defaults={
            "callbacks": callbacks,
        }
    )  
   
    # we can always append after the fact
    cli.trainer.callbacks.append(CustomCallback())

    arguments = cli.config

    is_local = arguments["data"].local

    cli.trainer.logger = TensorBoardLogger(
        "artifacts",
        name="alpaca",
        version=arguments["run_name"],
    )

    return cli, checkpointer, is_local

if __name__ == "__main__":

    cli, checkpoint_callback, is_local = cli_main()

    model = cli.model

    cli.trainer.fit(model, cli.datamodule)

    is_llama_model = isinstance(model._model, Llama)

    cli.trainer.save_checkpoint("artifacts/final.ckpt")

    if is_local is False:

        print("Download tokenizer")
        tokenizer_name = (
            DataNames.GPT2_TOKENIZER if is_llama_model is False else DataNames.LLAMA_DATASET
        )

        download_model(
            ml_client=get_ml_client(),
            name=tokenizer_name,
            destination="artifacts/tokenizer", 
        )

    if is_llama_model:
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            "artifacts/tokenizer"
        )
    else:
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            "artifacts/tokenizer"
        )

    print("Creating a traced model")
    traced_model = create_traced_model(tokenizer, model._model) # Do it on the pt model
    traced_model.save("artifacts/traced.pt")

    if is_local is False:

        best_model_path = checkpoint_callback.best_model_path

        ml_client = get_ml_client()

        print("Register a traced model")
        file_model = Model(
            path="artifacts/traced.pt",
            type=AssetTypes.CUSTOM_MODEL,
            name="TEST_alpaca_traced",
            description="XLMR trained on twitter sentiment dataset. traced"
        )
        ml_client.models.create_or_update(file_model)

        print("Register model")
        file_model = Model(
            path=best_model_path,
            type=AssetTypes.CUSTOM_MODEL,
            name="TEST_alpaca",
            description="Llama trained on alpaca data"
        )
        ml_client.models.create_or_update(file_model)

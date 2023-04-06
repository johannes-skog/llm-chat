import subprocess
import sys
import os
import torch
from datasets import load_dataset
import argparse
from azureml.core.run import Run
from azureml.core.dataset import Dataset
from datasets import load_dataset, load_from_disk
from azureml.core.model import Model
from model import GeneratorModel
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from util import (
    get_ml_client,
    get_latest_data_version,
    download_dataset,
    download_model,
    DataNames,
    create_traced_model
)
import torchmetrics
from lightning.pytorch.cli import LightningCLI
import lightning.pytorch as pl
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import transformers
from model import Gpt2, Llama

def main():
    pass

class DataModule(pl.LightningDataModule):
    
    DATASET_PATH = f"artifacts/alpaca/downloaded"
    
    def __init__(
        self,
        dataset_name: str,
        batch_size_train: int = 1,
        workers: int = 2,
        download: bool = False,
        subset_train: int = None,
    ):
        
        super().__init__()
        
        if download:
            download_dataset(
                ml_client=get_ml_client(),
                name=dataset_name,
                destination=DataModule.DATASET_PATH,
            )
        
        self.dataset_name = dataset_name
        self.workers = workers
        self.subset_train = subset_train
        self.batch_size_train = batch_size_train

    def setup(self, stage: str):
        
        self._dataset_hg = load_from_disk(DataModule.DATASET_PATH)

        print(self._dataset_hg["train"]["input_ids"].max())

    def train_dataloader(self):
        
        dataset = self._dataset_hg["train"]

        if self.subset_train is not None:
            dataset = dataset.select(range(self.subset_train))
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        None

    def test_dataloader(self):
        None

    def predict_dataloader(self):
        None

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...

def cli_main():

    checkpoint_callback_spaced = ModelCheckpoint(
        every_n_train_steps=5000,
    )
    # Some bug in lightning, this needs to be set manually outside of
    # the constructor..complains aboout monitor not set otherwise
    checkpoint_callback_spaced.save_top_k = 2

    checkpoint_callback = ModelCheckpoint(
        save_on_train_epoch_end=True,
    )
    
    cli = LightningCLI(
        save_config_overwrite=True,
        model_class=GeneratorModel,
        trainer_class=pl.Trainer,
        datamodule_class=DataModule,
        run=False,
        trainer_defaults={
            "callbacks": [
                checkpoint_callback_spaced,
            ],
        }
    )

    cli.trainer.logger = TensorBoardLogger("artifacts", name="alpaca")

    cli.datamodule.setup("git")

    print(cli.trainer.logger.log_dir, cli.trainer.logger.name)

    return cli, checkpoint_callback_spaced

if __name__ == "__main__":

    cli, checkpoint_callback = cli_main()

    # print(cli.model._model._encoder.print_trainable_parameters())
    model = cli.model

    cli.trainer.fit(model, cli.datamodule)

    is_llama_model = isinstance(model._model, Llama)

    best_model_path = checkpoint_callback.best_model_path
    print(best_model_path)

    cli.trainer.save_checkpoint("artifacts/final.ckpt")

    """

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

    best_model_path = checkpoint_callback.best_model_path

    ml_client = get_ml_client()
    
    print("Register model")
    file_model = Model(
        path=best_model_path,
        type=AssetTypes.CUSTOM_MODEL,
        name="TEST_alpaca",
        description="Llama trained on alpaca data"
    )
    ml_client.models.create_or_update(file_model)

    print("Register a traced model")
    traced_model = create_traced_model(tokenizer, model._model) # Do it on the pt model
    traced_model.save("artifacts/traced.pt")
    file_model = Model(
        path="artifacts/traced.pt",
        type=AssetTypes.CUSTOM_MODEL,
        name="TEST_alpaca_traced",
        description="XLMR trained on twitter sentiment dataset. traced"
    )
    ml_client.models.create_or_update(file_model)
    
    """
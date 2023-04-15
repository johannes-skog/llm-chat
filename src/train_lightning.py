import os
import torch
from datasets import load_from_disk
from azureml.core.model import Model
from model import GeneratorModel
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.constants import AssetTypes
from lightning.pytorch.cli import LightningCLI
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import transformers
from lightning.pytorch.strategies import DeepSpeedStrategy
from model import Gpt2, Llama
import json
import yaml
from pathlib import Path
from util import (
    get_ml_client,
    download_dataset,
    download_model,
    DataNames,
    create_traced_model,
    AzureBlobStorageCheckpoint,
    AzureBlobStorage,
    remove,
)

import argparse

def setup_callbacks(save_every_n_steps: int = 2000):

    callbacks = []

    checkpoint_callback_spaced = ModelCheckpoint(
        every_n_train_steps=save_every_n_steps,
    )
    # Some bug in lightning, this needs to be set manually outside of
    # the constructor..complains aboout monitor not set otherwise
    # checkpoint_callback_spaced.save_top_k = 2

    callbacks.append(checkpoint_callback_spaced)

    # Save to Azure blob storage
    if False:

        checkpoint_az = AzureBlobStorageCheckpoint(
            azure_blob_storage=AzureBlobStorage(
                connection_string=os.getenv("AZURE_CHECKPOINT_CONNECTION_STRING"),
                container_name=os.getenv("AZURE_CHECKPOINT_CONTAINER_NAME"),
            ),
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


class DataModule(pl.LightningDataModule):
    
    DATASET_PATH = f"artifacts/dataset"
    
    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 1,
        workers: int = 2,
        subset_train: int = None,
        local: bool = False,
        include_val: bool = False,
    ):
        
        super().__init__()

        self.data_path = os.path.join(DataModule.DATASET_PATH, dataset_name)
        
        if local is False:

            download_dataset(
                ml_client=get_ml_client(),
                name=dataset_name,
                destination=self.data_path,
            )
        
        self.local = local
        self.dataset_name = dataset_name
        self.workers = workers
        self.subset_train = subset_train
        self.batch_size = batch_size
        self.include_val = include_val

    def setup(self, stage: str):
        
        self._dataset_hg = load_from_disk(self.data_path)
        if self.include_val is True:
            self._dataset_hg = self._dataset_hg["train"].train_test_split(test_size=0.01)

        return self

    def train_dataloader(self):
        
        dataset = self._dataset_hg["train"]

        if self.subset_train is not None:
            dataset = dataset.select(range(self.subset_train))
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        
        if self.include_val is False:
            return None
        
        dataset = self._dataset_hg["test"]# .select(range(8))

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
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

class CustomCallback(pl.callbacks.Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")

def cli_main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config', type=str, default="config.yaml",
        help='Path to the config file'
    )

    parser.add_argument(
        '--local',
        action='store_true',
        default=False,
        help="do not push any to the cloud"
    )

    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())

    deepspeed_config = config.get("deepspeed", None)

    if deepspeed_config is not None:
        
        strategy = DeepSpeedStrategy(
            config=json.loads(deepspeed_config)
        )   

        config["trainer"]["strategy"] = strategy

    if torch.cuda.is_available() is False:
        config["trainer"]["precision"] = 32 # we cant use 16 bit precision on cpu

    model = GeneratorModel(**config["model"])

    datamodule = DataModule(**config["data"]).setup("fit")

    callbacks, checkpointer = setup_callbacks(
        save_every_n_steps = config["save_every_n_steps"]
    )
    callbacks.append(CustomCallback())

    trainer = pl.Trainer(
        **config["trainer"],
        callbacks=callbacks,
        logger=TensorBoardLogger(
            **config["logger"],
        )
    )

    pl.utilities.seed.seed_everything(config["seed"])

    return trainer, model, datamodule, checkpointer, config, args

if __name__ == "__main__":

    trainer, model, datamodule, checkpointer, config, args = cli_main()

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=config["resume_from_checkpoint"]
    )

    print("Done training... saving model")

    save_path = os.path.join(
        f"{config['logger']['save_dir']}" +"/"+ f"{config['logger']['name']}",
        "final.ckpt"
    )

    remove(save_path)

    trainer.save_checkpoint(save_path)

    print("Saving model to {}".format(save_path))

    if args.local is False:

        print("Getting tokenizer from az....")
        print("Download tokenizer")
        download_model(
            ml_client=get_ml_client(),
            name=f"tokenizer_{config['model']['model']}",
            destination="artifacts/tokenizer", 
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "artifacts/tokenizer"
        )

    print("Loading tokenizer from local...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "artifacts/tokenizer"
    )

    best_model_path = checkpointer.best_model_path

    print("Best model path", best_model_path)

    if args.local is False:

        ml_client = get_ml_client()

        """
        print("Creating a traced model")
        traced_model = create_traced_model(tokenizer, model._model) # Do it on the pt model
        traced_model.save("artifacts/traced.pt")
        print("Register a traced model")
        file_model = Model(
            path="artifacts/traced.pt",
            type=AssetTypes.CUSTOM_MODEL,
            name="TEST_alpaca_traced",
            description="XLMR trained on twitter sentiment dataset. traced"
        )
        ml_client.models.create_or_update(file_model)
        """

        print("Register model")
        file_model = Model(
            path=best_model_path,
            type=AssetTypes.CUSTOM_MODEL,
            name="TEST_alpaca",
            description="Llama trained on alpaca data"
        )
        ml_client.models.create_or_update(file_model)

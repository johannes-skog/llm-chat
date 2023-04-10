
from azure.ai.ml import MLClient
from azureml.core.authentication import ServicePrincipalAuthentication
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils
import os
import torch

IGNORE_LOSS_ID = -100

class DataNames:

    GPT2_TOKENIZER = "gpt2_tokenizer"
    GPT2_DATASET = "gpt2_alpaca_data_cleaned"

    LLAMA_TOKENIZER = "llama_tokenizer"
    LLAMA_DATASET = "llama_alpaca_data_cleaned"


def get_latest_data_version(name: str):

    ml_client = get_ml_client()

    version = max(
        [int(m.version) for m in ml_client.data.list(name=name)]
    )

    return version

def get_latest_model_version(name: str):

    ml_client = get_ml_client()

    version = max(
        [int(m.version) for m in ml_client.models.list(name=name)]
    )

    return version

def get_ml_client():

    svc_pr = ServicePrincipalAuthentication(
        tenant_id=os.getenv('AZURE_TENANT_ID'),
        service_principal_id=os.getenv('AZURE_CLIENT_ID'),
        service_principal_password=os.getenv('AZURE_CLIENT_SECRET')
    )

    ml_client = MLClient(
        svc_pr,
        subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID'),
        resource_group_name=os.getenv('AZURE_RESOURCE_GROUP'),
        workspace_name=os.getenv('AZURE_WORKSPACE_NAME')
    )

    return ml_client

def download_dataset(
    ml_client,
    name: str,
    destination: str,
    version: str = None,
):
    if version is None:
        version = get_latest_data_version(name)
    
    print("version is: " + str(version)  + " for " + name)
    data_info = ml_client.data.get(name=name, version=str(version))

    artifact_utils.download_artifact_from_aml_uri(
        uri=data_info.path,
        destination=destination,
        datastore_operation=ml_client.datastores
    )

def download_model(
    ml_client,
    name: str,
    destination: str,
    version: str = None,
):
    if version is None:
        version = get_latest_model_version(name)

    print("version is: " + str(version)  + " for " + name)
    data_info = ml_client.models.get(name=name, version=str(version))

    artifact_utils.download_artifact_from_aml_uri(
        uri=data_info.path,
        destination=destination,
        datastore_operation=ml_client.datastores
    )

def generate_prompt(input: str, instruction: str):
    
    if input != "":
        
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
             f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        )
        
    else:
        
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:"
        )

class TokenizerTokens:
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "</s>"
    DEFAULT_UNK_TOKEN = "</s>"


def create_traced_model(tokenizer, model):

    dd = tokenizer(
        ["This is a test...", "Detta är ett test..."],
        return_tensors="pt",
        padding=True
    )

    model.eval().cpu()

    jit_model = torch.jit.trace(
        model.forward,
        example_kwarg_inputs={
            "input_ids": dd["input_ids"],
            "attention_mask": dd["attention_mask"]
        }
    )

    return jit_model


from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
def unshard_deepspeed(save_path: str, output_path: str = "final.ckpt"):
    # lightning deepspeed has saved a directory instead of a file, shareded model
    # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html#deepspeed-zero-stage-3-single-file
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
    # Only the LORA-params are saved to final.ckpt -> non strict load from checkpoint after the initial weights have been loaded



import os
from typing import Optional
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from lightning.pytorch.callbacks import ModelCheckpoint

class AzureBlobStorageCheckpoint(ModelCheckpoint):

    def __init__(self, connection_string: str, container_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    def push_to_az(self, filepath: str) -> None:
        # Upload the saved checkpoint to Azure Blob Storage
        blob_name = os.path.relpath(filepath, self.dirpath)
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_name)
        
        print("pushing to azure", filepath, " as ", blob_name)

        with open(filepath, "rb") as data:
            blob_client.upload_blob(data)

    def _save_checkpoint(self, trainer: str, filepath: str) -> None:
        # Save model locally
        trainer.save_checkpoint(filepath, self.save_weights_only)
        
        self.push_to_az(filepath)

    def download_blob(self, blob: str, destination: str):

        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob)
        with open(destination, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

    def download_latest_checkpoint(self, checkpoints_dir: str) -> Optional[str]:
        # Create ContainerClient
        container_client = self.blob_service_client.get_container_client(self.container_name)

        # List all blobs in the container
        blobs = container_client.list_blobs()

        # Find the latest checkpoint based on the modification time
        latest_checkpoint = None
        latest_checkpoint_time = None
        for blob in blobs:
            if latest_checkpoint_time is None or blob.last_modified > latest_checkpoint_time:
                latest_checkpoint = blob
                latest_checkpoint_time = blob.last_modified

        if latest_checkpoint is None:
            return None

        # Download the latest checkpoint
        os.makedirs(checkpoints_dir, exist_ok=True)

        local_checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint.name)

        self.download_blob(
            blob=latest_checkpoint.name,
            destination=local_checkpoint_path
        )

        return local_checkpoint_path

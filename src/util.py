
from azure.ai.ml import MLClient
from azureml.core.authentication import ServicePrincipalAuthentication
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import os
import transformers
import torch
from typing import List
import os
from typing import Optional
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from lightning.pytorch.callbacks import ModelCheckpoint


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

def setup_tokenizer(name: str, max_tokens: int):

    if "lama" in name:

        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            name
        )

        tokenizer.add_special_tokens(
            {
                "eos_token": TokenizerTokens.DEFAULT_EOS_TOKEN,
                "bos_token": TokenizerTokens.DEFAULT_BOS_TOKEN,
                "unk_token": TokenizerTokens.DEFAULT_UNK_TOKEN,
                "pad_token": TokenizerTokens.DEFAULT_PAD_TOKEN,
            }
        )

    else:

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            name
        )

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = max_tokens

    return tokenizer

def create_traced_model(tokenizer, model, device: str = "cpu"):

    dd = tokenizer(
        ["This is a test...", "Detta är ett test..."],
        return_tensors="pt",
        padding=True
    )

    model.eval().to(device)

    jit_model = torch.jit.trace(
        model.forward,
        example_kwarg_inputs={
            "input_ids": dd["input_ids"].to(device),
            "attention_mask": dd["attention_mask"].to(device)
        }
    )

    return jit_model

def unshard_deepspeed(save_path: str, output_path: str = "final.ckpt"):
    # lightning deepspeed has saved a directory instead of a file, shareded model
    # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html#deepspeed-zero-stage-3-single-file
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
    # Only the LORA-params are saved to final.ckpt -> non strict load from checkpoint after the initial weights have been loaded

class AzureBlobStorage(object):

    def __init__(self, connection_string: str, container_name: str):

        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

    def upload_file(self, filepath: str, blob_name: str) -> None:

        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )

        with open(filepath, "rb") as data:
            blob_client.upload_blob(data)

    def download_blob(self, blob: str, destination: str):

        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob
        )

        with open(destination, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

    def get_latest_blobs(self) -> List[str]:

        container_client = self.blob_service_client.get_container_client(
            self.container_name
        )

        blobs = container_client.list_blobs()

        sorted_blobs = sorted(blobs, key=lambda x: x.last_modified, reverse=True)

        return [blob.name for blob in sorted_blobs]


class AzureBlobStorageCheckpoint(ModelCheckpoint):

    def __init__(self, azure_blob_storage: AzureBlobStorage, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.azure_blob_storage = azure_blob_storage

    def _save_checkpoint(self, trainer, filepath) -> None:

        trainer.save_checkpoint(filepath, self.save_weights_only)
        blob_name = os.path.relpath(filepath, self.dirpath)
        self.azure_blob_storage.upload_file(filepath, blob_name)

    def download_latest_checkpoint(self, checkpoints_dir: str) -> Optional[str]:

        latest_checkpoints = self.azure_blob_storage.get_latest_blobs()

        if not latest_checkpoints:
            return None

        latest_checkpoint_name = latest_checkpoints[0]

        os.makedirs(checkpoints_dir, exist_ok=True)
        local_checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint_name)
        self.azure_blob_storage.download_blob(latest_checkpoint_name, local_checkpoint_path)

        return local_checkpoint_path
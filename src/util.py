
from azure.ai.ml import MLClient
from azureml.core.authentication import ServicePrincipalAuthentication
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils
import os
import torch

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
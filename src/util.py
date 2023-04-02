
from azure.ai.ml import MLClient
from azureml.core.authentication import ServicePrincipalAuthentication
import os



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
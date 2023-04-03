from datasets import load_dataset
import argparse
from azure.ai.ml.entities import Model
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from util import get_ml_client, generate_prompt, TokenizerTokens
import transformers
import shutil

def _setup_dataset(tokenizer, path: str, destination: str):

    dataset = load_dataset('json', data_files=path)

    dataset = dataset.map(
        lambda e: {
            "promt_input": generate_prompt(
                input=e["input"],
                instruction=e["instruction"]
            )
        }
    )

    dataset = dataset.map(
        lambda e: {"input_ids": tokenizer.encode(
                e["promt_input"],
                max_length=tokenizer.model_max_length,
                truncation=True,
                padding='max_length',
            )
        }
    )

    dataset = dataset.map(
        lambda e: {"target_ids": tokenizer.encode(
                e["output"],
                max_length=tokenizer.model_max_length,
                truncation=True,
                padding='max_length',
            )
        }
    )

    dataset.set_format(
        type='torch',
        columns=['input_ids', 'target_ids']
    )

    dataset = dataset.map(
        lambda e: {
            "attention_mask": e["input_ids"].ne(tokenizer.pad_token_id)
        }
    )

    dataset = dataset.map(
        lambda e: {
                "target_weight": e["target_ids"].ne(tokenizer.pad_token_id).float()
        }
    )

    dataset.set_format(
        type='torch',
        columns=['input_ids', 'target_ids', "attention_mask", "target_weight"]
    )

    dataset.save_to_disk(destination)

    return dataset
   
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_name', type=str, default="alpaca_data_cleaned",
        help="name of the dataset as will be called in az"
    )

    parser.add_argument(
        '--max_tokens', type=int, default=512,
        help="The maximum number of tokens the model will be trained on."
    )

    parser.add_argument(
        '--model', type=str, default='llama',
    )

    args = parser.parse_args()

    assert args.model in ['llama', 'gpt2'], "only llama and gpt2 are supported"

    ml_client = get_ml_client()

    if args.model == "llama":
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            "decapoda-research/llama-7b-hf"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-large")

    tokenizer.add_special_tokens(
        {
            "eos_token": TokenizerTokens.DEFAULT_EOS_TOKEN,
            "bos_token": TokenizerTokens.DEFAULT_BOS_TOKEN,
            "unk_token": TokenizerTokens.DEFAULT_UNK_TOKEN,
            "pad_token": TokenizerTokens.DEFAULT_PAD_TOKEN,
        }
    )

    tokenizer.model_max_length = args.max_tokens

    tokenizer_path = f"artifacts/tokenizer/{args.model}" 
    dataset_path = f"artifacts/dataset/{args.model}" 

    print("Register the tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    file_model = Model(
        path=tokenizer_path,
        type=AssetTypes.CUSTOM_MODEL,
        name=f"{args.model}_tokenizer" ,
        description=f"{args.model} tokenizer for hg",
    )
    ml_client.models.create_or_update(file_model)

    print("working with the dataset")
    dataset = _setup_dataset(
        tokenizer=tokenizer,
        path="data/alpaca_data_cleaned.json",
        destination=dataset_path,
    )

    dataset = Data(
        path=dataset_path,
        type=AssetTypes.URI_FOLDER,
        description=f"alpaca data cleaned for {args.model}",
        name=f"{args.model}_alpaca_data_cleaned",
    )
    ml_client.data.create_or_update(dataset)

if __name__ == "__main__":
    main()
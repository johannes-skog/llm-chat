from datasets import load_dataset
import argparse
from azure.ai.ml.entities import Model
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from util import get_ml_client, generate_prompt, TokenizerTokens
import transformers
from util import DataNames
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

    def set_labels(example):
        target_ids = example["target_ids"]
        target_ids[target_ids==tokenizer.pad_token_id] = -100 # ignore loss on padding tokens
        example["target_ids"] = target_ids
        
        return example

    dataset = dataset.map(set_labels)

    dataset.save_to_disk(destination)

    return dataset
   
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--max_tokens', type=int, default=512,
        help="The maximum number of tokens the model will be trained on."
    )

    parser.add_argument(
        '--data_file', type=str,
        help="path to the json-file containing the data",
        default="data/alpaca_data_cleaned.json",
    )

    parser.add_argument(
        '--model', type=str, default='llama',
    )

    parser.add_argument(
        '--local',
        action='store_true',
        default=False,
        help="do not push any to the cloud"
    )

    args = parser.parse_args()

    assert args.model in ['llama', 'gpt2'], "only llama and gpt2 are supported"

    if args.model == "llama":
        
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            "decapoda-research/llama-7b-hf"
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

        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2-large")
        # add_special_tokens for gpt2 tokenizer is out of vocabulary -> added token... embedding will crash
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = args.max_tokens

    tokenizer_path = f"artifacts/tokenizer" 
    dataset_path = f"artifacts/dataset" 

    print("working with the dataset")
    dataset = _setup_dataset(
        tokenizer=tokenizer,
        path=args.data_file,
        destination=dataset_path,
    )

    tokenizer.save_pretrained(tokenizer_path)

    if args.local is False:

        ml_client = get_ml_client()

        print("Register the tokenizer")
        file_model = Model(
            path=tokenizer_path,
            type=AssetTypes.CUSTOM_MODEL,
            name=DataNames.GPT2_TOKENIZER if args.model == "gpt2" else DataNames.LLAMA_TOKENIZER ,
            description=f"{args.model} tokenizer for hg",
        )
        ml_client.models.create_or_update(file_model)

        dataset = Data(
            path=dataset_path,
            type=AssetTypes.URI_FOLDER,
            description=f"alpaca data cleaned for {args.model}",
            name=DataNames.GPT2_DATASET if args.model == "gpt2" else DataNames.LLAMA_DATASET,
        )
        ml_client.data.create_or_update(dataset)

if __name__ == "__main__":
    main()
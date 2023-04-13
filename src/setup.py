from datasets import load_dataset
import argparse
from azure.ai.ml.entities import Model
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from util import get_ml_client, generate_prompt, setup_tokenizer
import transformers
from util import DataNames, IGNORE_LOSS_ID
import copy 
from transformers import AutoTokenizer

def _setup_dataset(tokenizer, path: str, destination: str, debug: bool = False):

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
        lambda e: {"input_ids_source": tokenizer.encode(
                e["promt_input"],
                max_length=tokenizer.model_max_length,
                truncation=True,
                padding='max_length',
            )
        }
    )

    dataset = dataset.map(
        lambda e: {"target": e["promt_input"] + e["output"] + tokenizer.eos_token}  
    )

    dataset = dataset.map(
        lambda e: {"target_ids": tokenizer.encode(
                e["target"],
                max_length=tokenizer.model_max_length,
                truncation=True,
                padding='max_length',
            )
        }
    )

    dataset.set_format(type='torch', columns=["target_ids", "input_ids_source"])

    def _process_targets(e):

        len_source = e["input_ids_source"].ne(tokenizer.pad_token_id).sum().item()

        e["target_ids_masked"] = copy.deepcopy(
            e["target_ids"]
        )

        e["attention_mask"] = e["target_ids_masked"].ne(tokenizer.pad_token_id).long()

        e["target_ids_masked"][0:len_source] = IGNORE_LOSS_ID

        e["target_ids_masked"][
            e["target_ids_masked"]==tokenizer.pad_token_id
        ] = IGNORE_LOSS_ID

        return e

    dataset = dataset.map(
        lambda e: _process_targets(e)
    )

    if debug is False:

        dataset = dataset.remove_columns(
            ["instruction", "input", "output", "promt_input", "target", "target_ids"]
        )
        dataset = dataset.rename_column("target_ids_masked", "target_ids")
        dataset = dataset.rename_column("input_ids_source", "input_ids")

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
        '--local',
        action='store_true',
        default=False,
        help="do not push any to the cloud"
    )

    parser.add_argument(
        '--model',
        default="decapoda-research/llama-7b-hf",
        help="which model to use"
    )

    args = parser.parse_args()

    tokenizer = setup_tokenizer(args.model, max_tokens=args.max_tokens) 

    tokenizer_path = f"artifacts/tokenizer/{args.model}" 
    dataset_path = f"artifacts/dataset/{args.model}" 

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
            name=f"tokenizer_{args.model}",
            description=f"{args.model} tokenizer for hg",
        )
        ml_client.models.create_or_update(file_model)

        dataset = Data(
            path=dataset_path,
            type=AssetTypes.URI_FOLDER,
            description=f"alpaca data cleaned for {args.model}",
            name=f"{args.model}",
        )
        ml_client.data.create_or_update(dataset)

    print("done")

if __name__ == "__main__":
    main()
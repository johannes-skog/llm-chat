from datasets import load_dataset
import argparse
from azure.ai.ml.entities import Model
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from util import get_ml_client, generate_prompt, TokenizerTokens
import transformers


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

    dataset.set_format(type='torch', columns=['input_ids', 'target_ids'])

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

    args = parser.parse_args()

    ml_client = get_ml_client()

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

    tokenizer.model_max_length = args.max_tokens

    print("Register the tokenizer")
    tokenizer.save_pretrained("artifacts/tokenizer")
    file_model = Model(
        path="artifacts/tokenizer",
        type=AssetTypes.CUSTOM_MODEL,
        name="llama_tokenizer",
        description="Llama tokenizer for hg"
    )
    ml_client.models.create_or_update(file_model)

    print("working with the dataset")
    dataset = _setup_dataset(
        tokenizer=tokenizer,
        path="data/alpaca_data_cleaned.json",
        destination="artifacts/dataset",
    )

    dataset = Data(
        path="artifacts/dataset",
        type=AssetTypes.URI_FOLDER,
        description="alpaca data cleaned",
        name=args.dataset_name,
    )
    ml_client.data.create_or_update(dataset)

if __name__ == "__main__":
    main()
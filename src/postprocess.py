from datasets import load_dataset
import argparse
from azure.ai.ml.entities import Model
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from util import get_ml_client, generate_prompt, TokenizerTokens
import transformers
from util import DataNames, IGNORE_LOSS_ID
import copy 
from pathlib import Path
import yaml
from model import GeneratorModel
import torch

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_file', type=str,
        help="path to the json-file containing the data",
        default="data/alpaca_data_cleaned.json",
    )

    parser.add_argument(
        '--model', type=str, default="decapoda-research/llama-7b-hf",
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        help="where is the checkpoint located?"
    )

    parser.add_argument(
        '--destination',
        type=str,
        default="artifacts/processed_model",
        help="where is the checkpoint saved to?"
    )

    parser.add_argument(
        '--training_config',
        type=str,
        default="config.yaml",
        help="did we use deepspeed when training the model?"
    )

    parser.add_argument(
        '--local',
        action='store_true',
        default=False,
        help="do not push any to the cloud"
    )

    args = parser.parse_args()

    config = yaml.safe_load(Path(args.training_conf).read_text())

    deepspeed_training = (
        "deepspeed" in config["trainer"]["strategy"]["class_path"].lower() if isinstance(config["trainer"]["strategy"], dict) 
        else "deepspeed" in config["trainer"]["strategy"]    
    )

    if deepspeed_training is True:
        from util import unshard_deepspeed
        unshard_deepspeed(args.checkpoint_path, "artifacts/deepspeed_unshared.ckpt")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "artifacts/tokenizer"
    )

    model = GeneratorModel(**config["model"])

    d = {k: v for k, v in torch.load("final.ckpt")["state_dict"].items() if "lora" in k}
    model.load_state_dict(d, strict=False)

    model.save_pretrained(args.destination)

if __name__ == "__main__":
    main()
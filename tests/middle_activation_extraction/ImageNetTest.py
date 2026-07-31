import torch
import torch.nn as nn

import toml
from easydict import EasyDict
from tqdm import tqdm
import argparse
from src.SAEs.evaluate.EvalTokenExtraction import ImageNetTestTokenExtraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024, help="提取的批次大小")
    parser_args = parser.parse_args()
    config_dict = toml.load('config.toml')
    args = EasyDict(config_dict)
    ImageNetTestTokenExtraction(args, batch_size=parser_args.batch_size)


if __name__ == "__main__":
    main()
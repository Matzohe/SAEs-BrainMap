import os
import torch
import numpy as np
import argparse
from easydict import EasyDict

def set_seed(seed: int):
    
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_path(path):
    path = "/".join(path.split("/")[ :-1])
    if not os.path.exists(path):
        os.makedirs(path)


def check_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        try:
            data = torch.tensor(data)
            return data
        except ValueError:
            print("Current data type not support to transform to torch.Tensor")

def get_info_from_shell(arg_parse: argparse.Namespace, 
                        args: EasyDict
                    ) -> EasyDict:
    """
    从shell中读取信息，然后将其放入到args中。函数返回EasyDict的args
    由于EasyDict的args是两层的结构，所以如果arg_parse中的参数不含_，则不输入args

    Args:
        arg_parse (argparse.Namespace): 从命令行中获得的相关新参数
        args (EasyDict): 默认参数，如果arg_parse中有新的参数，那么arg_parse中的参数会覆盖args
    
    Returns: 
        EasyDict: args
    """

    for key, value in vars(arg_parse).items():
        if value is None or "_" not in key:
            continue
        key1 = key.split("_")[0]
        key2 = "_".join(key.split("_")[1:])
        if key1 in args and key2 in args[key1]:
                args[key1][key2] = value

    return args

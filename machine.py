"""utility functionality that aids in successfully and optimally setting up an enviroment for models to run on, based on the originating machine."""

import torch
from typing import Literal


def get_optimal_device() -> Literal["cpu","cuda"]:
    """select the optimal supported device available on the running machine"""
    if  torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
    

def get_optimal_compute_type() -> Literal["int8","float16","float32"]:
    """select the optimal compute type available on the running machine"""
    #implement
    return "float32"
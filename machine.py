"""utility functionality that aids in successfully and optimally setting up an enviroment for models to run on, based on the originating machine."""

import torch
from typing import Literal
import gc


### SETUP CONFIGURATION ###

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


### RUNTIME OPTIMISATION ###

def clear_gpu() -> None:
    """free up space within the GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    


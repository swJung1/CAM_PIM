import copy
import random
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm

import torch
import copy
import torch.nn as nn
import itertools
import pandas as pd

from qlayercam_backup_1020 import *
from utils.dataloader import *

assert torch.cuda.is_available(), \
"The current runtime does not have CUDA support." \
"Please go to menu bar (Runtime - Change runtime type) and select GPU"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

    
def load_model(model_name):
    if model_name == 'resnet20':
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True) # pretrained cifar10 model load
    return model


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        # self.input = input
        self.weight = module.weight
        # self.output = output
        
        if hasattr(module, "bitWeight"):
            weight2D = module.weight_2d_mapping(self.weight)
            weightQ, _, _ = LinearQuantizeW(weight2D, sum(self.bitWeight), weight2D.max(), weight2D.min())
        
        weightQM = weightQ // (2.**module.bitWeightLSB)
        weightQL = weightQ % (2.**module.bitWeightLSB)
        
        numSubArray = int(weight2D.shape[1]/module.subArray)
        numSubRow = [self.subArray] * numSubArray + ([] if weight2D.shape[1] % module.subArray == 0 else [weight2D.shape[1] % module.subArray])
        for s, rowArray in enumerate(numSubRow):
            mask = torch.zeros_like(weight2D)
            mask[:,(s*module.subArray):(s+1)*module.subArray] = 1
            
            subweight = weight2D * mask
            print(subweight)
            
            
    def close(self):
        self.hook.remove()


bitWeight = 8
bitWeightLSB = 4
hid_subarray = [64, 128, 256, 100_0000]


c_model = []
for subArray in hid_subarray:
    model = load_model('resnet20')
    hook_list = []
    sl = {}
    
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and 'downsample' not in name and m.in_channels != 3:
            weight2D = m.weight.reshape(m.weight.shape[0], -1)
            weightQ, _, _ = LinearQuantizeW(weight2D, bitWeight, weight2D.max(), weight2D.min())
            
            weightQM = weightQ // (2.*bitWeightLSB)
            weightQL = weightQ % (2.*bitWeightLSB)
            
            numSubArray = int(weight2D.shape[1]/subArray)
            numSubRow = [subArray] * numSubArray + ([] if weight2D.shape[1] % subArray == 0 else [weight2D.shape[1] % subArray])
            
            uniqs = {name: 0 for name in range(weightQM.shape[1])}
            for s, rowArray in enumerate(numSubRow):
                mask = torch.zeros_like(weight2D)
                mask[:,(s*subArray):(s+1)*subArray] = 1
                
                weightQMS = weightQM * mask
                tt = []
                # weightQLS = weightQL * mask
                for N in range(weightQMS.shape[1]):
                    tt[N] = len(torch.unique(weightQMS[:,N]))
                    
                    for name, val in tt.items():
                        if uniqs[name] < val:
                            uniqs[name] = val
            print(uniqs)
        
        
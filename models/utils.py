import copy
import torch.nn as nn


def clone_module(module, k):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

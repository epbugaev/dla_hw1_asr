import torch
from torch import Tensor, nn
from torchvision.transforms.v2 import RandomApply
import torchaudio.transforms

class FrequencyMasking(nn.Module):
    def __init__(self, p=0.2, freq_mask_param=10, *args, **kwargs):
        super().__init__()
        self._aug = RandomApply(nn.ModuleList([torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param, *args, **kwargs)]), p=p)

    def __call__(self, data: Tensor):
        x = data#.unsqueeze(1)
        device = x.device
        return self._aug(x).to(device)#.squeeze(1)
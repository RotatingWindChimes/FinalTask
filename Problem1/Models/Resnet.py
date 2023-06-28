import torch
import torch.nn as nn
from  torchvision.models import resnet18

class Resnet_pretrain(nn.Module):
    def __init__(self,pretrain):
        super().__init__()
        self.base_encoder = resnet18(num_classes=128)
        self.base_encoder.load_state_dict(torch.load(pretrain))
        for para in self.base_encoder.parameters():
            para.requires_grad = False
        self.classification = nn.Linear(128,100)

    def forward(self,x):
        x = self.base_encoder(x)
        x = nn.functional.normalize(x, dim=1)
        return self.classification(x)

class Resnet_full(nn.Module):
    def __init__(self,pretrain):
        super().__init__()
        self.base_encoder = resnet18(num_classes=128)
        self.base_encoder.load_state_dict(torch.load(pretrain))
        self.classification = nn.Linear(128,100)

    def forward(self,x):
        x = self.base_encoder(x)
        x = nn.functional.normalize(x, dim=1)
        return self.classification(x)

class Resnet_normal(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_encoder = resnet18(num_classes=128)
        self.classification = nn.Linear(128,100)


    def forward(self,x):
        x = self.base_encoder(x)
        x = nn.functional.normalize(x, dim=1)
        return self.classification(x)
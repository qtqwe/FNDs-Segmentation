import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import os

new_torch_home = "classify_model"

os.environ["TORCH_HOME"] = new_torch_home


class CustomViT(nn.Module):
    def __init__(self, num_classes: int = 8):
        super(CustomViT, self).__init__()
        # 使用预训练的ViT模型
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)
        self.dropout = nn.Dropout(0.5)
        self.vit.heads = nn.Sequential(
            nn.Linear(self.vit.heads[0].in_features, self.vit.heads[0].in_features),
            self.dropout,
            nn.Linear(self.vit.heads[0].in_features, num_classes)
        )

    def forward(self, x):
        return self.vit(x)

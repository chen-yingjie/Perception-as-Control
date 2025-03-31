from typing import Tuple
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention import AdaLayerNorm, Attention, FeedForward

from src.models.motion_module import zero_module
from src.models.resnet import InflatedConv3d


class FusionModule(ModelMixin):
    def __init__(
        self,
        in_channels: int = 640,
        out_channels: int = 320,
        fusion_type: str = 'conv',
        fusion_with_norm: bool = True,
    ):
        super().__init__()

        self.fusion_type = fusion_type
        self.fusion_with_norm = fusion_with_norm
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        if self.fusion_type == 'sum':
            self.fusion = None
        elif self.fusion_type == 'max':
            self.fusion = None
        elif self.fusion_type == 'conv':
            self.fusion = InflatedConv3d(
                in_channels, out_channels, kernel_size=3, padding=1
            )
        
    def forward(self, feat1, feat2):
        b, c, t, h, w = feat1.shape

        if self.fusion_type == 'sum':
            return feat1 + feat2
        elif self.fusion_type == 'max':
            return torch.max(feat1, feat2)
        elif self.fusion_type == 'conv':
            b, c, t, h, w = feat1.shape
            if self.fusion_with_norm:
                feat1 = rearrange(feat1, "b c t h w -> (b t) (h w) c")
                feat2 = rearrange(feat2, "b c t h w -> (b t) (h w) c")
                feat1 = self.norm1(feat1)
                feat2 = self.norm2(feat2)
                feat1 = feat1.view(b, t, h, w, c)
                feat2 = feat2.view(b, t, h, w, c)
                feat1 = feat1.permute(0, 4, 1, 2, 3).contiguous()
                feat2 = feat2.permute(0, 4, 1, 2, 3).contiguous()
            feat = torch.concat((feat1, feat2), 1)
            feat = self.fusion(feat)
            
        return feat
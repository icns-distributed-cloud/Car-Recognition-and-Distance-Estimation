from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module

import math

class ROIPool(nn.Module):
    def __init__(self, output_size):
        super(ROIPool, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size)
        self.size = output_size
    

    def forward(self, feature_map, bbox):   
        _, _, h, w = feature_map.shape    

        roi = feature_map[:, :, math.floor(h*bbox[1]):math.ceil(h*bbox[3])+1, math.floor(w*bbox[0]):math.ceil(w*bbox[2])+1]
        roi = self.maxpool(roi)
        roi = roi.view(1, -1).cuda()

        return roi






        


        
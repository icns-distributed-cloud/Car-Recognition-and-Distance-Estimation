from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module




class Dist(nn.Module):
    def __init__(self):
        super(Dist, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

        self.softplus = nn.Softplus()


    def forward(self, rois):
        output = self.fc1(rois)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.softplus(output)

        return output



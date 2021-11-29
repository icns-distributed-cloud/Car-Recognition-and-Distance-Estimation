from utils.datasets import *
from roipool import *
from model_dist import *

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from PIL import Image
import cv2
import time

if __name__ == '__main__':
    # print('hello')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg16 = torchvision.models.vgg16(pretrained=True).to(device)
    # print('hi')
    feature_extractor = vgg16.features
    feature_extractor.eval()
    
    for param in feature_extractor.parameters():
        param.requires_grad = False

    roipool = ROIPool((2, 2)).to(device)
    roipool.eval()

    model_distance = Dist().to(device)

    dataset = ListDataset('original_data/train.txt')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model_distance.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma= 0.1)  

    for epoch in range(2000):
        model_distance.train()
        total_loss = 0
        for batch_i, (img_path, imgs, targets, distance) in enumerate(dataloader):
            # print(feature_extractor[0].bias)
            imgs = imgs.to(device)
            bboxes = targets.to(device)
            distance = distance.float().to(device)

            optimizer.zero_grad()

            feature_map = feature_extractor(imgs)
            
            for i in range(len(bboxes[0])):
                roi = roipool(feature_map, bboxes[0][i])
                output = model_distance(roi)
                target = distance[0][i]
                target = target.unsqueeze(0)
                
                loss = loss_fn(output, target)
                
                loss.backward()
            optimizer.step()
        scheduler.step()

                # print(f'loss = {loss}')
        print(f'epoch = {epoch}')
        print(model_distance.fc1.bias)

        if epoch != 0:
            torch.save(model_distance.state_dict(), f'checkpoints_distance5/tiny1_{epoch}.pth')

        if epoch % 2 == 0 and epoch != 0:
            model_distance.eval()

            total_loss = 0
            num = 0
            for batch_i, (img_path, imgs, targets, distance) in enumerate(dataloader):
                imgs = imgs.to(device)
                bboxes = targets.to(device)
                distance = distance.float().to(device)
                
                feature_map = feature_extractor(imgs)
                for i in range(len(bboxes)):
                    roi = roipool(feature_map, bboxes[0][i])
                    output = model_distance(roi)
                    target = distance[0][i]
                    target = target.unsqueeze(0)

                    loss = (output, target) ** 2
                    total_loss += loss
                    num += 1
            
            rmse = (total_loss/num)**(0.5)
            print(f'RMSE: {rmse}')
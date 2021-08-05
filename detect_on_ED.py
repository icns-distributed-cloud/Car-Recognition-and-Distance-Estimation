from utils.datasets import *
from roipool import *
from model_dist import *

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

import cv2
import datetime

def changeRGB2BGR(img):
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r

    return img


if __name__ == '__main__':

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    dummy_img = torch.zeros((1, 3, 416, 416)).float()

    model = torchvision.models.vgg16(pretrained=True).to(device)
    model.eval()
    feature_extractor = model.features
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
    
    roipool = ROIPool((2, 2)).to(device)
    roipool.eval()
    model_distance = Dist().to(device)
    cc = datetime.datetime.now()
    model_distance.load_state_dict(torch.load('checkpoints_distance5/tiny1_156.pth', map_location=device))
    dd = datetime.datetime.now()
    print(dd-cc)
    model_distance.eval()
    
    dataset = ListDataset('original_data/valid.txt', multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    total_loss = 0
    num = 0
    # model_distance.train()
    
    for batch_i, (img_path, imgs, targets, distance) in enumerate(dataloader):
        imgs = imgs.to(device)

        bboxes = targets.to(device)
        distance = distance.to(device)

        feature_list = []
        actual_list = []
        output_distance = []
        aa = datetime.datetime.now()
        with torch.no_grad():
            feature_map = feature_extractor(imgs)
            for bbox in bboxes[0]:
                roi = roipool(feature_map, bbox).to(device)
                feature_list.append(roi)
            
            for i in range(len(feature_list)):
                output = model_distance(feature_list[i])
        
                target = distance[0][i]
                output = output.squeeze(0)
                actual_list.append(target)
                output_distance.append(output)

        img = cv2.imread(img_path[0])
        bb = datetime.datetime.now()
    
        print(bb - aa)
    
    

    


   
    
    


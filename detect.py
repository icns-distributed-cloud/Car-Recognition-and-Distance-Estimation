from utils.datasets import *
from roipool import *
from model_dist import *

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

import cv2


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_img = torch.zeros((1, 3, 416, 416)).float()

    model = torchvision.models.vgg16(pretrained=True).to(device)
    model.eval()
    feature_extractor = model.features
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
    
    roipool = ROIPool((2, 2)).to(device)
    roipool.eval()
    model_distance = Dist().to(device)
    
    model_distance.load_state_dict(torch.load('checkpoints_distance4/tiny1_40.pth'))
    model_distance.eval()
    
    dataset = ListDataset('original_data/train.txt', multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    total_loss = 0
    num = 0
    model_distance.train()
    for batch_i, (img_path, imgs, targets, distance) in enumerate(dataloader):
        imgs = imgs.to(device)

        bboxes = targets.to(device)
        distance = distance.to(device)

        feature_list = []
        with torch.no_grad():
            feature_map = feature_extractor(imgs)
            for bbox in bboxes[0]:
                roi = roipool(feature_map, bbox)
                feature_list.append(roi)
            
            for i in range(len(feature_list)):
                output = model_distance(feature_list[i])

                target = distance[0][i]
                output = output.squeeze(0)
                
                
                print(f'output = {output}   target = {target}')
    #             actual_list.append(target)
    #             output_distance.append(output)
    #             loss = (output-target)**2
    #             total_loss += loss
    #             num += 1
    #     img = cv2.imread(img_path[0])
    #     bounding_box = targets_distance[0][:,:-1]
    #     idx = 0
    #     for box in bounding_box:
    #         x1 = (box[0]-(box[2]/2))*640
    #         x2 = (box[0]+(box[2]/2))*640
    #         y1 = (box[1]-(box[3]/2))*480
    #         y2 = (box[1]+(box[3]/2))*480
    #         img = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0,), 2)
    #         distan = '%.2f' % output_distance[idx]
    #         actual = '%.2f' % actual_list[idx]
    #         cv2.putText(img, (f'predited = {distan}'), (x1,y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    #         cv2.putText(img, (f'actual = {actual}'), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    #         idx += 1

    #     # img = cv2.UMat(changeRGB2BGR(imgs[0].cpu().numpy().transpose(1,2,0)))
    #     # print(type(img))
    #     # cv2.putText(img, str(1), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
 
    #     cv2.imshow(f'{img_path}', img)

    #     key = cv2.waitKey(0) & 0xFF
    #     if key == ord('q'):
    #         cv2.destroyWindow(f'{img_path}')
    #         continue

    #     elif key == ord('s'):
    #         break

    # print(f'loss = {total_loss/num}')

    
    

    


   
    
    


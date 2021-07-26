from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

import os
import numpy as np
from PIL import Image



class ListDataset(Dataset):
    def __init__(self, list_path, augment=False, multiscale=False, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path, 'r').convert('RGB'))


        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution


        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        targets_distance = None
        if os.path.exists(label_path):
            if torch.from_numpy(np.loadtxt(label_path)).ndim == 2:
                boxes = torch.from_numpy(np.loadtxt(label_path)[:,4:8].reshape(-1, 4))
                boxes[:,0] = boxes[:,0]/w
                boxes[:,1] = boxes[:,1]/h
                boxes[:,2] = boxes[:,2]/w
                boxes[:,3] = boxes[:,3]/h
            else:
                boxes = torch.from_numpy(np.loadtxt(label_path)[4:8].reshape(-1, 4))
                boxes[:,0] = boxes[:,0]/w
                boxes[:,1] = boxes[:,1]/h
                boxes[:,2] = boxes[:,2]/w
                boxes[:,3] = boxes[:,3]/h
            # Extract coordinates for unpadded + unscaled image

            

            targets = torch.zeros((len(boxes), 4))
            targets = boxes

            if torch.from_numpy(np.loadtxt(label_path)).ndim == 2:
                targets_distance = torch.from_numpy(np.loadtxt(label_path)[:,13].reshape(-1, 1))
            else:
                targets_distance = torch.from_numpy(np.loadtxt(label_path)[13].reshape(-1, 1))
            
        # Apply augmentations
        # if self.augment:
        #    if np.random.random() < 0.5:
        #        img, targets = horisontal_flip(img, targets)
        
        return img_path, img, targets, targets_distance

    def __len__(self):
        return len(self.img_files)
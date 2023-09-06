import torch
from torchvision import transforms
import numpy as np

class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform = None, classes=None):
        self.dataset = dataset
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {class_label: class_idx for class_idx, class_label in enumerate(classes)}
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx) :
        img, label = self.dataset[idx]
        # class_name = self.classes[label]
        if self.transform :
            img = self.transform(img)
        return img, label
        
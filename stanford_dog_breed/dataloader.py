import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from hyperparameters import *
from custom_dataset import *
import os

data_dir = './dog_images'
subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]

dog_dataset = ImageFolder(root=data_dir)

NoT = int(len(dog_dataset)*TRAIN_RATIO)
NoTest = int(len(dog_dataset)*TEST_RATIO)
NoV = len(dog_dataset) - NoT - NoTest

dog_dataset_train, dog_dataset_val ,dog_dataset_test = torch.utils.data.random_split(dog_dataset, [NoT, NoV, NoTest])

transform_train = transforms.Compose([
    
                                transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET, interpolation=transforms.InterpolationMode.BILINEAR),     # AutoAugment

                                transforms.AugMix(),
                                transforms.ToTensor(),
                                transforms.Resize((224, 224), antialias=True),
                                transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
                                transforms.RandomHorizontalFlip(),   # 수평 뒤집기
                                transforms.RandomRotation(30),        # 랜덤한 각도로 회전 (±30도 범위)
                                # transforms.RandomResizedCrop(224, antialias=True, scale=(0.2, 0.2)),    # 랜덤 크롭 및 리사이즈 (224x224 크기),
                                # transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # 스케일링 (0.8배에서 1.2배 크기로)
                                # transforms.RandomAffine(degrees=0, shear=(-10, 10)),   # 원근 변환 (최대 ±10도)
                                # transforms.ColorJitter(brightness=0.5),                # 밝기 조절 (최대 ±0.5)
                                # transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # 랜덤한 이동, 회전, 크기 조절
                                # transforms.ColorJitter(saturation=0.5),         # 채도 조절 (최대 ±0.5)
                                # transforms.ColorJitter(hue=0.2),                # 색상 조절 (최대 ±0.2)
                                # transforms.ColorJitter(contrast=0.5),           # 명암 대비 조절 (최대 ±0.5)
                                ])

transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((224, 224), antialias=True),
                                transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
                                ])

class_names = dog_dataset.classes

train_DS = Custom_Dataset(dog_dataset_train, transform=transform_train, classes=class_names)
val_DS = Custom_Dataset(dog_dataset_val, transform=transform_test, classes=class_names)
test_DS = Custom_Dataset(dog_dataset_test, transform=transform_test, classes=class_names)


train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True, num_workers = 2)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=True, num_workers = 2)
val_DL = torch.utils.data.DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=True, num_workers = 2)



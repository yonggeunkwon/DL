# import
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn, optim
from earlystopping import *
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from resnet import *
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.001
EPOCH = 3
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
criterion = nn.CrossEntropyLoss()
new_model_train = True
save_model_path = f"../stanford_dog_breed/Resnet50_dog_breed_3.pt"

img = Image.open('./dog_images/n02085620-Chihuahua/n02085620_10074.jpg')
img.size

def Train(model, train_DL, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    loss_history = []
    
    for inputs, labels in train_DL:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        y_hat = model(inputs)
        loss = criterion(y_hat, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(y_hat.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    average_loss = total_loss / len(train_DL)
    accuracy = correct / total * 100
    loss_history.append(average_loss)  # Loss 저장
    return loss_history, accuracy, average_loss

def Test(model, test_DL):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_DL:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    return accuracy

# Validation 함수 정의
def VAL(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    average_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    return average_loss, accuracy, average_loss

def Test_plot(model, test_DL):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(test_DL))
        x_batch = x_batch.to(DEVICE)
        y_hat = model(x_batch)
        pred = y_hat.argmax(dim=1)

    x_batch = x_batch.to('cpu')

    plt.figure(figsize=(8, 4))
    for idx in range(6):
        plt.subplot(2, 3, idx+1, xticks = [], yticks = [])
        plt.imshow(x_batch[idx].permute(1, 2, 0).squeeze(), cmap="gray")
        pred_class = test_DL.dataset.classes[pred[idx]]
        true_class = test_DL.dataset.classes[y_batch[idx]]
        plt.title(f"{pred_class} ({true_class})", color = 'g' if pred_class == true_class else 'r')

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num


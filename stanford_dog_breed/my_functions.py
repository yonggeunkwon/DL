# import
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
import time
import pandas as pd
DEVICE = 'cuda'if torch.cuda.is_available() else 'cpu'


# Validation 함수 정의
# def VAL(model, dataloader, criterion):
#     model.eval()
#     total_loss = 0.0
#     correct = 0
#     total = 0
    
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
#             outputs = model(inputs)
#             inputs = inputs.to(DEVICE)
#             labels = labels.to(DEVICE)
#             loss = criterion(outputs, labels)
            
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
    
#     average_loss = total_loss / len(dataloader)
#     accuracy = correct / total * 100
#     return accuracy, average_loss

def Test_plot(model, test_DL):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(test_DL))
        x_batch = x_batch.to(DEVICE)
        y_hat = model(x_batch)
        pred = y_hat.argmax(dim=1)

    x_batch = x_batch.to('cpu')

    plt.figure(figsize=(12, 8))
    for idx in range(6):
        plt.subplot(2, 3, idx+1, xticks = [], yticks = [])
        plt.imshow(x_batch[idx].permute(1, 2, 0).squeeze(), cmap="gray")
        pred_class = test_DL.dataset.classes[pred[idx]]
        true_class = test_DL.dataset.classes[y_batch[idx]]
        plt.title(f"{pred_class} ({true_class})", color = 'g' if pred_class == true_class else 'r')
    plt.savefig('test_plot.png') 


def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num


import torch
import pandas as pd

def calculate_class_accuracy(model, test_DL):
    model.eval()
    num_classes = len(test_DL.dataset.classes)
    
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    class_actual = [0] * num_classes
    
    with torch.no_grad():
        for x_batch, y_batch in test_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            y_hat = model(x_batch)
            pred = y_hat.argmax(dim=1)
            
            correct_preds = pred == y_batch
            for i, class_label in enumerate(test_DL.dataset.classes):
                class_index = test_DL.dataset.class_to_idx[class_label]
                class_correct[class_index] += correct_preds[y_batch == i].sum().item()
                class_total[class_index] += (y_batch == i).sum().item()
                class_actual[class_index] += (pred == i).sum().item()
    
    class_accuracies = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
    
    class_data = {'Class': test_DL.dataset.classes, 'Class_Count': class_total, 'Predicted_Count': class_actual, 'Correct_Count': class_correct, 'Accuracy': class_accuracies}
    return pd.DataFrame(class_data)





# def Train(model, train_DL, criterion, optimizer,
#            EPOCH, BATCH_SIZE, TRAIN_RATIO, save_model_path, save_history_path, is_scheduler = False):
#     loss_history = []
#     NoT = len(train_DL.dataset) # 60000

#     if is_scheduler:
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0)
#     else:
#         scheduler = None

#     loss_history = {'train':[], 'val':[]}
#     acc_history = {'train':[], 'val':[]}
#     best_loss = 9999

#     for ep in range(EPOCH):
#         epoch_start = time.time()
#         current_lr = optimizer.param_groups[0]['lr']
#         print(f"Epoch: {ep+1}, current_LR = {current_lr}")

#         model.train() # train mode로 전환    
#         train_loss, train_accuracy, _ = loss_epoch(model, train_DL, criterion, optimizer)
#         loss_history['train'] += [train_loss]
#         acc_history['train'] += [train_accuracy]

#         model.eval() # test mode로 전환
#         with torch.no_grad():
#             val_loss, val_accuracy, _ = loss_epoch(model, train_DL, criterion)
#             loss_history['val'] += [val_loss]
#             acc_history['val'] += [val_accuracy]

#             if val_loss < best_loss:
#                 best_loss = val_loss
#                 torch.save({'model' : model,
#                             'ep':ep,
#                             'optimizer' : optimizer}, 
#                            save_model_path)

#         if is_scheduler:
#             scheduler.step()

#         # print loss
#         print(f"train loss: {round(train_loss,5)}, "
#               f"val loss: {round(val_loss,5)} \n"
#               f"train acc: {round(train_accuracy,1)} %, "
#               f"val acc: {round(val_accuracy,1)} %, time: {round(time.time()-epoch_start)} s")
#         print("-"*20)

#     torch.save({"loss_history": loss_history,
#                 "acc_history": acc_history,
#                 "EPOCH": EPOCH,
#                 "BATCH_SIZE": BATCH_SIZE,
#                 "TRAIN_RATIO": TRAIN_RATIO}, save_history_path)

#     return loss_history

# def Test(model, test_DL, criterion):
#     model.eval()
#     with torch.no_grad():
#         test_loss, test_accuracy, rcorrect = loss_epoch(model, test_DL, criterion, optimizer=None)
#     print(f"Test accuracy: {rcorrect}/{len(test_DL.dataset)} ({round(test_accuracy,1)} %)")
#     return round(test_accuracy, 1)


# Validation 함수 정의
def VAL(model, dataloader, criterion):
    model.eval()
    # total_loss = 0
    correct = 0
    total = 0
    r_loss = 0.0
    misclassified_images = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            r_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 틀린 예측을 식별하고 이미지를 리스트에 추가
            misclassified_mask = (predicted != labels)
            misclassified_batch = inputs[misclassified_mask]
            misclassified_labels = predicted[misclassified_mask]
            for i in range(len(misclassified_batch)):
                misclassified_images.append((misclassified_batch[i], misclassified_labels[i]))
    
        loss_e = r_loss / len(dataloader.dataset)
        accuracy = correct / total * 100
    return accuracy, loss_e, misclassified_images

def Test(model, dataloader, criterion):
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
    return accuracy, average_loss

def Train(model, train_DL, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    loss_b = 0
    total = 0
    rloss = 0
    # loss_history = []
    
    for inputs, labels in train_DL:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        y_hat = model(inputs)
        loss = criterion(y_hat, labels)

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        rloss += loss_b

        _, predicted = torch.max(y_hat.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    loss_e = total_loss / len(train_DL)
    accuracy = correct / total * 100
    # loss_history.append(average_loss)  # Loss 저장/
    return accuracy, loss_e


# def loss_epoch(model, DL, criterion, optimizer = None):
#     N = len(DL.dataset) # The number of data
#     rloss = 0 # running loss
#     rcorrect = 0
#     for x_batch, y_batch in DL:
#         x_batch = x_batch.to(DEVICE)
#         y_batch = y_batch.to(DEVICE)
#         # inference
#         y_hat = model(x_batch)
#         # loss
#         loss = criterion(y_hat, y_batch)
#         # update
#         if optimizer is not None:
#             optimizer.zero_grad() # gradient 누적을 막기 위한 초기회
#             loss.backward() # backpropagation
#             optimizer.step() # weight update
#         # loss accumulation
#         loss_b = loss.item() * x_batch.shape[0] # batch loss # BATCH_SIZE를 곱하면 마지막 18개도 32개를 곱하게 된다.
#         rloss += loss_b
#         # accuracy accumulation
#         pred = y_hat.argmax(dim=1)
#         corrects_b = torch.sum(pred == y_batch).item()
#         rcorrect += corrects_b
#     loss_e = rloss/N
#     accuracy_e = rcorrect/N*100

#     return loss_e, accuracy_e, rcorrect
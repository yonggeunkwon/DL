from model import *
from my_functions import *
from dataloader import *
from torch import nn, optim
from wandb import *
import torchvision
import wandb
import custom_scheduler

# wandb
project_name = 'Dog Classification'
group_name = 'Grad CAM'
wandb.init(
    project=project_name,
    group=group_name
    )
            
args = {
    "learning_rate": LR,
    "epochs": EPOCH,
    "batch_size": BATCH_SIZE,
    "optimizer": OPTIMIZER,
    "weight_decay": weight_decay
}

wandb.config.update(args)

# 실행 이름 설정
wandb.run.name = 'AugMix + Rotation + Flip'
wandb.run.save()

model = ResNet50().to(DEVICE)
optimizer = OPTIMIZER(model.parameters(), lr=LR, weight_decay=weight_decay)


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


scheduler = custom_scheduler.CosineAnnealingWarmUpRestarts(
    optimizer, T_0=50, T_mult=1, eta_max=0.001,  T_up=10, gamma=0.5)


train_loss_history = []
val_loss_history = []

for epoch in range(EPOCH):
    train_accuracy, train_loss_e = Train(model, train_DL, criterion, optimizer)
    test_accuracy, _ = Test(model, test_DL, criterion)
    valid_accuracy, val_loss_e, misclassified_images = VAL(model, val_DL, criterion)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch: {epoch+1}, current_LR = {current_lr}")
    print(f"Epoch [{epoch+1}/{EPOCH}] - Train Loss: {train_loss_e:.4f} - Train Accuracy: {train_accuracy:.2f}% - Test Accuracy: {test_accuracy:.2f}%")    
    print(f"Validation Loss: {val_loss_e:.4f} - Validation Accuracy: {valid_accuracy:.2f}%")
    print("-" * 20)
    
    train_loss_history += [train_loss_e]
    val_loss_history += [val_loss_e]

    wandb.log({"Training loss": train_loss_e, 
               "val loss": val_loss_e,
               'Training Acc': train_accuracy,
               'Val Acc': valid_accuracy,
               'Test Acc': test_accuracy,
               "Learning Rate" : current_lr
               })

    # Early stopping
    if early_stop == True:
        if val_loss_e < best_val_loss_e:
            best_val_loss_e = val_loss_e
            best_val_accuracy = valid_accuracy
            best_train_accuracy = train_accuracy
            best_test_accuracy = test_accuracy
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f'Early stop : Validation loss didn’t improve for {patience} epochs.')
                print(f'Best Accuracy : Train : {best_train_accuracy:.2f}, Valid : {best_val_accuracy:.2f}, Test : {best_test_accuracy:.2f}')
                break
    
    # Scheduler
    # scheduler.step()
    
torch.save(model.state_dict(), save_model_path)

# 틀린 이미지 저장
misclassified_dir = 'aug_rotation_flip_misclassified_images'
if not os.path.exists(misclassified_dir):
    os.makedirs(misclassified_dir)

for i, (image, label) in enumerate(misclassified_images):
    save_path = os.path.join(misclassified_dir, f'misclassified_{i}.png')
    torchvision.utils.save_image(image, save_path)


class_accuracy_df = calculate_class_accuracy(model, test_DL)
class_accuracy_df.to_csv('class_accuracies.csv', index=False)    

plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label = 'train loss')
plt.plot(range(1, len(val_loss_history)+1), val_loss_history, label = 'val loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.title('Train, Val Loss')
plt.grid()
plt.savefig('resnet50.png')
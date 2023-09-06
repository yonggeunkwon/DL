from my_functions import *
from dataloader import *
from model import *
from torchvision import models
DEVICE = 'cuda'if torch.cuda.is_available() else 'cpu'
import wandb
wandb.init(project='Dog Classification')
# 실행 이름 설정
wandb.run.name = 'Pretrained Resnet 101 (freeze 9layers, lr=0.001) -1'
wandb.run.save()

args = {
    "learning_rate": LR,
    "epochs": EPOCH,
    "batch_size": BATCH_SIZE,
    "optimzer" : OPTIMIZER,
    "weight_decay" : weight_decay
}
wandb.config.update(args)

BATCH_SIZE = 64
LR = 0.001
EPOCH = 30
OPTIMIZER = optim.Adam
criterion = nn.CrossEntropyLoss()
pretrained_resnet = models.resnet101(models.ResNet101_Weights.IMAGENET1K_V2)
model = PretrainedResNet(pretrained_resnet, 120, freeze=True).to(DEVICE)
optimizer = OPTIMIZER(model.parameters(), lr=LR, weight_decay=0.00001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0)
'---------------------------------------------'

# training
train_loss_history = []
val_loss_history = []

for epoch in range(EPOCH):
    train_accuracy, train_loss_e = Train(model, train_DL, criterion, optimizer)
    test_accuracy, _ = Test(model, test_DL, criterion)
    valid_accuracy, val_loss_e = VAL(model, val_DL, criterion)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch: {epoch+1}, current_LR = {current_lr}")
    print(f"Epoch [{epoch+1}/{EPOCH}] - Train Loss: {train_loss_e:.4f} - Train Accuracy: {train_accuracy:.2f}% - Test Accuracy: {test_accuracy:.2f}%")    
    print(f"Validation Loss: {val_loss_e:.4f} - Validation Accuracy: {valid_accuracy:.2f}%")
    print("-" * 20)
    
    train_loss_history += [train_loss_e]
    val_loss_history += [val_loss_e]

    wandb.log({"Training loss": train_loss_e, "val loss": val_loss_e, 'Training Acc': train_accuracy, 'Val Acc': valid_accuracy, 'Test Acc': test_accuracy})

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

plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label = 'train loss')
plt.plot(range(1, len(val_loss_history)+1), val_loss_history, label = 'val loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.title('Train, Val Loss')
plt.grid()
plt.savefig('pretrained.png')
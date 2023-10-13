from model import *
from my_functions import *
from dataloader import *
from torch import nn, optim
from wandb import *
import wandb

hyperparameter_defaults = dict(
    BATCH_SIZE = 32,
    EPOCH = 100,
    LR = 0.001,
    )

# wandb
project_name = 'Dog Classification'
group_name = 'Sweep'
wandb.init(
    project=project_name,
    group=group_name,
    config=hyperparameter_defaults
    )
config = wandb.config

print(config['BATCH_SIZE'])
print(config['LR'])
print(config['EPOCH'])

config_batch_size = config['BATCH_SIZE']
config_lr = config['LR']
config_epoch = config['EPOCH']

train_DL = torch.utils.data.DataLoader(train_DS, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers = 2)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers = 2)
val_DL = torch.utils.data.DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=True, num_workers = 2)

# 실행 이름 설정
wandb.run.name = f'LR : {config_lr}, Batch Size : {config_batch_size}, Epoch : {config_epoch}'
wandb.run.save()

model = ResNet50().to(DEVICE)
wandb.watch(model)
optimizer = OPTIMIZER(model.parameters(), lr=config['LR'], weight_decay=weight_decay)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

train_loss_history = []
val_loss_history = []

for epoch in range(config['EPOCH']):

    args = {
        "learning_rate": config['LR'],
        "epochs": config['EPOCH'],
        "batch_size": config['BATCH_SIZE'],
        "optimizer": OPTIMIZER,
        "weight_decay": weight_decay
    }

    wandb.config.update(args)

    train_accuracy, train_loss_e = Train(model, train_DL, criterion, optimizer)
    test_accuracy, _ = Test(model, test_DL, criterion)
    valid_accuracy, val_loss_e = VAL(model, val_DL, criterion)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch: {epoch+1}, current_LR = {current_lr}")
    print(f"Epoch [{epoch+1}/{config['EPOCH']}] - Train Loss: {train_loss_e:.4f} - Train Accuracy: {train_accuracy:.2f}% - Test Accuracy: {test_accuracy:.2f}%")    
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
plt.savefig('resnet50.png')
from torch import nn, optim

TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
VALID_RATIO = 0.1
BATCH_SIZE = 128
LR = 0.001
EPOCH = 100
weight_decay = 0.00001
l1_lambda = 0.01
early_stop = False
best_val_loss_e = float('inf')
patience = 5
OPTIMIZER = optim.Adam
criterion = nn.CrossEntropyLoss()
save_model_path = f"../stanford_dog_breed/aug_rotation_flip.pt"
save_history_path = f"../stanford_dog_breed/"
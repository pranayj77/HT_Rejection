from Models.models import *
from Datasets.datasets import *
import time 
import torch
from tqdm import tqdm
import copy
import sys


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device= torch.device('cuda')):
    since = time()
    val_acc_history = []
    train_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model = model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs_1,inputs_2,labels_1, labels_2,time_dif in tqdm(dataloaders[phase]):
                inputs_1 = inputs_1.to(device=device, dtype=torch.float)
                inputs_2 = inputs_2.to(device=device, dtype=torch.float)
                labels_1 = labels_1.to(device=device, dtype=torch.float)
                labels_2 = labels_2.to(device=device, dtype=torch.float)
                time_dif = time_dif.to(device=device, dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss                    
                    outputs = model(inputs_1,inputs_2,labels_1,time_dif)
                    loss = criterion(outputs, labels_2.unsqueeze(1))

                    preds = torch.squeeze(torch.round(torch.sigmoid(outputs)))
                    # print(preds.shape)
                    # print(labels.shape)
                    # print(preds == labels.data.T)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item() * inputs_1.size(0)
                running_corrects += torch.sum(preds == labels_2.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_acc_history.append(epoch_acc)

        print()

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history

if __name__ == '__main__':
    modelArg = sys.argv[1]

    if modelArg == '2.0':
        mod = TwinNet()
        dataMod = 'append'
    elif modelArg == '2.1':
        mod = TwinNet()
        dataMod = 'grad'
    
    train_data = PTBDataset(mode=dataMod)
    train_data = train_data.set_fold("Train")
    val_data = PTBDataset(mode=dataMod)
    val_data = val_data.set_fold("Val")
    test_data = PTBDataset(mode=dataMod)
    test_data = test_data.set_fold("Test")
    batch_size = 64
    trainLoader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, num_workers=10)

    valLoader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                            shuffle=True, num_workers=10)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(mod.parameters(),lr=0.005,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)
    dataloaders = {'train':trainLoader,'val':valLoader}

    mod, val_acc_history, train_acc_history = train_model(mod, dataloaders, criterion, optimizer, num_epochs=100, device= torch.device('cuda'))

    fname = "1d_CNN_siamese_100fs_Nov_3_with_time_lr_0.005"
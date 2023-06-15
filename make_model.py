# reference
# https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a

# import using Modules
import os
import time
import torchvision
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
from torch import nn
from torchvision import transforms,  datasets
from torch.utils.data import DataLoader, random_split
from setting import *

# to fix below ERROR
# #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

no_cuda = False

use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

##################
# train method      # TODO move to another python file
##################

def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=100, device='cpu'):

    print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
        (type(model).__name__, type(optimizer).__name__,
        optimizer.param_groups[0]['lr'], epochs, device))

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

    start_time_sec = time.time()
    for epoch in range(1, epochs+1):
        epoch_start = time.time()
        # --- train training set -----------------------------
        model.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0

        for idx, batch in enumerate(train_dl):

            optimizer.zero_grad()

            data, target = batch[0].to(device), batch[1].to(device)
            pred = model(data)
            loss = loss_fn(pred, target)

            loss.backward()
            optimizer.step()

            prec_cpu = pred.cpu()
            target_cpu = target.cpu()

            train_loss         += loss.data.item() * data.size(0)
            num_train_correct  += (torch.max(prec_cpu, 1)[1] == target_cpu).sum().item()
            num_train_examples += data.shape[0]

            if idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, idx * len(data), len(train_dl.dataset),
                    100. * idx / len(train_dl), loss.item()))

        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)


        # --- evaluate vlidation set -------------------------------------
        model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0

        for idx, batch in enumerate(val_dl):

            data, target = batch[0].to(device), batch[1].to(device)
            pred = model(data)
            prec_cpu = pred.cpu()
            target_cpu = target.cpu()
            loss = loss_fn(pred, target)

            prec_cpu = pred.cpu()
            target_cpu = target.cpu()

            val_loss         += loss.data.item() * data.size(0)
            num_val_correct  += (torch.max(prec_cpu, 1)[1] == target_cpu).sum().item()
            num_val_examples += target_cpu.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)


        if epoch > 0:
            print('Epoch time: [%3.f sec], %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
                    (time.time() - epoch_start, epoch, epochs, train_loss, train_acc, val_loss, val_acc))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    # end training


    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#################
# test method # TODO have to move another python file
#################
def test(model, loss_fn, test_dl, classes):
    model.eval()
    test_loss = 0
    correct = 0
    falsePositive = 0
    falseNegative = 0
    num_test_examples = 0
    with torch.no_grad():
        for idx, batch in enumerate(test_dl):
            data, target = batch[0].to(device), batch[1].to(device)
            pred = model(data)
            loss = loss_fn(pred, target)

            test_loss += loss.data.item() * data.size(0)
            cur_correct = (torch.max(pred, 1)[1] == target).sum().item()
            correct += cur_correct
            if cur_correct != 16:
                pred_name = [classes[i] for i in torch.max(pred, 1)[1].tolist()]
                target_name = [classes[i] for i in target.tolist()]
                print("pred: {}".format(pred_name))
                print("targ: {}".format(target_name))
                imshow(torchvision.utils.make_grid(data.cpu()))

            
            p = np.array(torch.max(pred.cpu(), 1)[1])
            r = np.array(target.cpu())
            falsePositive += ((p == 1) & (r == 0)).sum().item()
            falseNegative += ((p == 0) & (r == 1)).sum().item() 
            num_test_examples += data.shape[0]

    test_loss /= len(test_dl.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%) Recall: {:.3f}‰, Precision: {:.3f}‰\n'.format
        (test_loss, correct, num_test_examples,
        100. * correct / num_test_examples,
        100. * correct / (falsePositive + correct), 
        100. * correct / (falseNegative + correct)))

def check_Image(path):
    if os.path.getsize(path) > 0 and os.path.splitext(path)[1] != '.db':
        return True
    print(path)
    return False

def get_dataset_from_dir(path:str, data_transforms):
    whole_data = datasets.ImageFolder(path, transform = data_transforms, is_valid_file=check_Image)

    return whole_data

def split_dataset(whole_data, train_size = 0.7, valid_size = 0.2, test_size = 0.1):
    num_train = len(whole_data)

    valid_size = 0.20
    test_size = 0.10
    train_size = 0.70
    split1 = int(np.floor(valid_size * num_train))
    split2 = int(np.floor(test_size * num_train)) + split1
    print(split1, split2)

    print((int(np.floor(valid_size * num_train)), int(np.floor(test_size * num_train)), int(np.floor(train_size * num_train))))
    valid_set, test_set, train_set = random_split(whole_data, (int(np.round(valid_size * num_train)), int(np.round(test_size * num_train)), int(np.ceil(train_size * num_train))))
    print(len(valid_set), len(test_set), len(train_set))

    return train_set, valid_set, test_set
    
def dataset2dataloader(valid_set, test_set, train_set):
    training_loader = DataLoader(train_set, batch_size = 16, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size = 16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size = 16, shuffle=False)
    print(len(test_loader.dataset), len(valid_loader.dataset), len(training_loader.dataset))

    return training_loader, valid_loader, test_loader


def show_batch_image(whole_data, loader):
    idx2class = {v: k for k, v in whole_data.class_to_idx.items()}
    g = 1
    rows = 4
    cols = 4
    figure = plt.figure(figsize=(16, 16))
    matplotlib.rcParams['font.family'] ='Malgun Gothic'
    matplotlib.rcParams['axes.unicode_minus'] =False
    for i,j in loader:   
        figure.add_subplot(rows, cols, g)
        for img in i:        
            idx = j[0].item()
            lbl = idx2class[idx]
            img = img.swapaxes(0, 1)
            img = img.swapaxes(1, 2)
            plt.title(lbl)
            plt.imshow(img, cmap="gray", interpolation='nearest', aspect='auto')        
            break
        g += 1    
        if g > 16:
            break
    plt.show()

def get_model(out_class):
    model_trained = torchvision.models.resnet101(pretrained=True)
    num_ftrs = model_trained.fc.in_features
    model_trained.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1024),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.Dropout(0.1),
        nn.Linear(512, out_class),
        #nn.Sigmoid()
    )
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    model_trained.to(device)

    return model_trained


if __name__ == "__main__":

# transfoms for images resizing and normarization/regulatation
    #subset data
    data_dir = '.\\picked_classes'

    transform = transforms.Compose([
                                        transforms.ToTensor(),
                                    ])

    whole_data = get_dataset_from_dir(data_dir, transform)


    train_set, valid_set, test_set = split_dataset(whole_data)
    train_loader, valid_loader, test_loader = dataset2dataloader(train_set, valid_set, test_set)

    model = get_model(54)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    if TRAIN_FLAG:
        history = train(model, optimizer,criterion, train_loader,valid_loader, epochs=20, device=device)
                    
        test(model, criterion, test_loader)

    if SAVE_MODEL:
        torch.save(model, './model.pth')
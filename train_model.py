#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import argparse
import json
import logging
import os
import sys


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms, models

try:
    import smdebug.pytorch as smd
except:
    pass

try:
    from smdebug import modes
except:
    pass

try:
    from smdebug.profiler.utils import str2bool
except:
    pass

try:
    from smdebug.pytorch import get_hook
except:
    pass

try:
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except:
    pass

def test(model, test_loader, criterion,device,hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")


def train(model, train_loader, valid_loader, criterion, optimizer,device,epochs,hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':valid_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0
            hook.set_mode(smd.modes.TRAIN)
            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                accuracy = running_corrects/running_samples
                print("Images [{}/{} ({:.0f}%)] , Test set: Average loss: {:.4f} Accuracy: {}/{} ({:.2f}%)".format(
                        running_samples,
                        len(image_dataset[phase].dataset),
                        100.0 * (running_samples / len(image_dataset[phase].dataset)),
                        loss.item(),
                        running_corrects,
                        running_samples,
                        100.0*accuracy,
                    )
                )
            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

        if loss_counter==1:
            break
    return model
    
def net():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 10))
    return model

def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, 'dogbreedmodel04feb.pt'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model

def main(args):
    epochs       = args.epochs
    lr           = args.lr
    batch_size   = args.batch_size
    test_batch_size   = args.test_batch_size    
    momentum     = args.momentum  
    model_dir = args.model_dir
    data_dir = args.data_dir
    
    model=net()
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])    
    
    trainset = datasets.ImageFolder(root=os.path.join(data_dir,'train'), transform=training_transform)
    
    validset = datasets.ImageFolder(root=os.path.join(data_dir,'valid'), transform=training_transform)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        shuffle=True)
    
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
        shuffle=True)
    
    model=model.to(device)
    
    model=train(model=model, train_loader=train_loader, valid_loader=valid_loader, criterion=loss_criterion, optimizer=optimizer,device=device,epochs=epochs,hook=hook)
    
    del train_loader
    del valid_loader
    
    '''
    TODO: Test the model to see its accuracy
    '''
    testing_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])   

    testset = datasets.ImageFolder(root=os.path.join(data_dir,'test'), transform=training_transform)

    test_loader = torch.utils.data.DataLoader(trainset, batch_size=test_batch_size,
        shuffle=True)
    
    test(model=model, test_loader=test_loader, criterion=loss_criterion,device=device,hook=hook)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, os.path.join(model_dir,'dogbreedmodel04feb.pt'))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    
    main(args)

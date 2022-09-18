#---------------------------------------------------------------------
#CSYE7105 - Final Project - Spring 2022
#Team 2
#Contributors: Pramod Gopal, Pratiksha Patole
#Project Title: Performance Analysis of Multiple GPUs for a Scene Recognition Computer Vision Task
#This code is for training scene recognition task using Distributed Data Paralell method using multi GPUs with simple ConvNet architecture
#---------------------------------------------------------------------

import numpy as np 
import pandas as pd 
import PIL
from PIL import Image
import os
import json
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from apex import amp


data_dir = "/home/gopal.p/pramod_gopal_csye7105/final_project/datset_v2/places365/train"
PRINT_LOGS = False

classes = []

for directory in os.listdir(data_dir):
    if "." not in directory:           
        classes.append(directory)
        
if PRINT_LOGS == True:
    print('Classes: ', classes)

def get_data():
  image_paths = []
  image_classes = np.array([[cls]*250 for cls in classes]).reshape(-1) 
  
  for cls in classes:
      # adds the path of the image to the first column
      image_paths.extend(os.path.join(data_dir, cls, i) for i in os.listdir(os.path.join(data_dir, cls))) 
  #print(image_paths)
  print(len(image_paths))
  print(len(image_classes))
  data = pd.DataFrame({'path':image_paths, 'class':image_classes, })
  
  return data

f = open("class_mapping.json")
str_to_int = json.load(f)

# Create the Torch Dataset object for our pandas dataframe 
class Places365Dataset(Dataset):
    def __init__(self, data, root_dir, transform=transforms.ToTensor()):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.data.iloc[idx, 0]
        image = Image.open(img_name)
        y_label = torch.tensor(str_to_int[self.data.iloc[idx, 1]])
        
        if self.transform:
            image = self.transform(image)
    
        return (image, y_label)

class CNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CNN, self).__init__()
        self.out_dim = out_dim
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, 8, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=1, padding=0),
            nn.ReLU()
        )
        
        self.q1 = nn.Linear(22*22*32, 64) # The 22*22 are the final dimensions of the "image"
        self.q2 = nn.Linear(64, out_dim)
    
    def forward(self, x):
        conv = self.convs(x)
        flat = conv.reshape(-1, 22*22*32)
        q1 = F.relu(self.q1(flat))
        q = self.q2(q1)
        return q

# Main function will take in some arguments and run the training function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    #train(0, args)
    args.world_size = args.gpus * args.nodes                
    os.environ['MASTER_ADDR'] = 'localhost'              
    os.environ['MASTER_PORT'] = '12360'                      
    mp.spawn(train, nprocs=args.gpus, args=(args,))  

# Train function

def train(gpu, args):
    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    )        

    torch.manual_seed(0)
    model = CNN(3,365) # 365 is number of output classes
    print('Using Simple Convnet')
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 1024
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
                                      
     
    data = get_data() 
    
    # Those are the transformations we will apply in our dataset
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
        ])
    
    dataset = Places365Dataset(
    data=data,
    root_dir=data_dir,
    transform=transform
    )
    
    train_set, test_set = torch.utils.data.random_split(dataset, [80000,11250])
    test_loader = DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_set,
    	num_replicas=args.world_size,
    	rank=rank
    )
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)


    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
           # print("image shape == ", images.shape)
            #print("labels shape ==",labels.shape)
            # Forward pass
            outputs = model(images)
            #print("outputs shape == ", outputs.shape)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    loss.item())
                   )
              
    if gpu >=0:
        print("Training complete in: " + str(datetime.now() - start))

if __name__ == '__main__':
    main()

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch.nn.functional as F

from models.yolov1 import Yolov1, BBBasedYolov1
from models.yololoss import YoloLoss
from data_loader.data_loader import Compose, TennisDataset, imgshow

from engine import train_one_epoch

import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    # Model
    # 1: YoloV1, 2: BBBasedYolov1
    parser.add_argument("--model", type=int, default=2)
    
    # Train or Inference
    parser.add_argument("--mode", type=str, default='train')
    
    # Hyperparameter
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    
    # utils
    parser.add_argument("--device", type=str, default='cpu')
    
    # Saved Loss Location
    parser.add_argument("--file_name_loss", type=str)
    
    # Saved Model Location
    parser.add_argument("--file_name", type=str)
    
    return parser


def main(args):
    # Device Utilization
    device = None
    if args.device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    
    if args.mode == 'train':
        # Load Model
        model = None
        if args.model == 1:
            model = Yolov1(split_size=7, num_boxes=2, num_classes=3).to(device)
        elif args.model == 2:
            model = BBBasedYolov1(split_size=7, num_boxes=2, num_classes=3).to(device)
        print('Load Model Complete')
        
        # Load Loss Function
        loss_fn = nn.YoloLoss().to(device)
        
        # Load Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Load Dataset
        transform = Compose([transforms.ToTensor(),])
        IMG_DIR = "data/object-detection.v4i.yolov5pytorch/train/images"
        LABEL_DIR = "data/object-detection.v4i.yolov5pytorch/train/labels"
        
        ds = TennisDataset(
            img_dir = IMG_DIR,
            label_dir = LABEL_DIR,
            transform = transform
        )
        print('Load Dataset Complete')
        
        # Load DataLoader
        train_dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        
        # Train
        loss_list = []
        print('====================')
        for epoch in range(args.epochs):
            loss = train_one_epoch(model, loss_fn, optimizer, train_dl, device, epoch+1, args.epochs)
            loss_list.append(loss)
            print('--------------------')
        np.save(os.path.join('saved/loss', args.file_name_loss), np.array(loss_list))
            
        # Save Model
        if args.file_name:
            try:
                torch.save(model.state_dict(), os.path.join('saved', args.file_name))
                print("Saving Model Success!")
            except:
                print("Saving Model Failed...")

    elif args.mode == 'test':
        # Load Trained Model
        model = None
        if args.model == 1:
            model = Yolov1(split_size=7, num_boxes=2, num_classes=3).to(device)
        elif args.model == 2:
            model = BBBasedYolov1(split_size=7, num_boxes=2, num_classes=3).to(device)
        
        model.load_state_dict(torch.load(os.path.join('saved', args.file_name)))
        model.eval()
        print('Load Model Complete')
        
    elif args.mode == 'inference':
        # Load Trained Model
        model = None
        if args.model == 1:
            model = Yolov1(split_size=7, num_boxes=2, num_classes=3).to(device)
        elif args.model == 2:
            model = BBBasedYolov1(split_size=7, num_boxes=2, num_classes=3).to(device)
        
        model.load_state_dict(torch.load(os.path.join('saved', args.file_name)))
        model.eval()
        print('Load Model Complete')
        
        cap = cv2.VideoCapture('data/딥러닝과제_테스트.mp4')
        
        transform = transforms.ToTensor()

        while(cap.isOpened()):
            ret, frame = cap.read()
            frame = cv2.resize(frame, dsize=(960, 540))
            x = transform(frame)
            x = x.reshape(-1, *x.shape)

            print(x.shape)
            pred = model(x)
            print(pred.shape)
            pred[0, ..., 20] = (pred[0,...,20] > 0.60)*1
            imgshow(x[0].detach().cpu(), pred[0].detach().cpu())
            cv2.waitKey(1)
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("Yolo training and evaluation Script", parents=[get_args_parser()])
    args = parser.parse_args()
    
    main(args)
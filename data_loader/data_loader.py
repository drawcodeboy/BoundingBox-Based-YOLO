import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, boxes = t(img), bboxes

        return img, bboxes

class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, S=7, B=2, C=3, width=448, height=448, length=None, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.label_path = os.listdir(label_dir)
        
        if length is not None:
            # 작동 테스트를 위한 데이터셋 크기 지정
            self.label_path = self.label_path[:length]
        
        self.transform = transform
        self.width, self.height = width, height
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.label_path)

    def __getitem__(self, index):
        data_path = self.label_path[index] # Label 파일 위치
        
        label_path = os.path.join(self.label_dir, data_path)
        # print(label_path)
        boxes = []
        with open(label_path) as f:
            for label in f.readlines(): # label 내용 불러오기
                # 여기서 x, y는 원래 좌표에 대해 448로 각각 나눈 값이다.
                class_label, x, y, width, height = [ # integer = label, float = x, y, width, height
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, (data_path[:-3] + 'jpg')) # 이미지 파일 위치
        # print(img_path)
        
        image = Image.open(img_path) # 이미지 불러오기
        image = image.resize((self.width, self.height)) # 사이즈 변경
        boxes = torch.tensor(boxes) # box -> tensor 타입으로 바꾸기


        if self.transform: # 이미지 사이즈 변경 -> tensor 타입으로 변경
            image, boxes = self.transform(image, boxes) # box 는 없어도 되도록 수정가능

        # 20 = Class label의 수
        # 5 = 1(Objectness Score; Confidence) + 4(Bounding Box Coordinates)
        # 5 * self.B = 한 셀에서 제안하는 Bounding Box의 수
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B)) #7x7x30 <= (class:20+(y/n:1+xywh:4)*2)

        for box in boxes: # boxes: 이미지 마다 Label 이 저장
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # 물체가 있는 cell 찾기
            i = int(self.S * x)
            j = int(self.S * y)

            x_cell = self.S * x - i
            y_cell = self.S * y - j

            if label_matrix[i, j, 3] == 0:

                box_coordinates = torch.tensor(
                    # dx, dy, dw, dh
                    [x_cell, y_cell, width, height]
                )

                label_matrix[i, j, self.C] = 1
                label_matrix[i, j, self.C+1:self.C+5] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix

def imgshow(img, box):
    # Target이 있는 index를 2차원에서 1차원으로 flatten하여 나타낸다.
    inx = torch.where(box[..., 3].view(-1) == 1)[0]

    index = []

    for i in inx:
        index.append(divmod(i.numpy(), 7)) # 몇 번째 cell에 있는지 구함

    # fig, ax = plt.subplots(figsize=(5, 5))
    img_ = img.permute(1, 2, 0).numpy().copy()
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    print(img.shape)
    print(box.shape)

    for _, (i, j) in enumerate(index):
        dx, dy, dw, dh = box[i, j, 4:8]

        x = dx*(img.shape[2]//7) + i*(img.shape[2]//7)
        y = dy*(img.shape[1]//7) + j*(img.shape[1]//7)
        w = int(dw*img.shape[2])
        h = int(dh*img.shape[1])

        xx = np.max((int(x-w/2), 0))
        yy = np.max((int(y-h/2), 0))
        
        print(xx, yy, w, h)

        '''
        ax.add_patch(
            patches.Circle(
                (x, y),
                edgecolor='red',
                fill=False
            )
        )

        ax.add_patch(
            patches.Rectangle(
                (xx, yy),
                w, h,
                edgecolor = 'red',
                fill=False
            )
        )'''
        
        img_ = cv2.rectangle(img_, (xx, yy), (xx+w, yy+h), color=(255, 0, 0), thickness=2)
    cv2.namedWindow("Yolo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Yolo", (img_.shape[1], img_.shape[0]))
    cv2.imshow('Yolo', img_)
    
if __name__ == '__main__':
    transform = Compose([transforms.ToTensor(),])
    IMG_DIR = "data/object-detection.v4i.yolov5pytorch/train/images"
    LABEL_DIR = "data/object-detection.v4i.yolov5pytorch/train/labels"
    ds = CustomDataset(
        img_dir = IMG_DIR,
        label_dir = LABEL_DIR,
        transform = transform,
        width=960,
        height=540
    )
    imgshow(ds[101][0], ds[101][1])
    cv2.waitKey(0)
    
    '''
    IMG_DIR = "data/Pascal VOC 2012.v1-raw.yolov5pytorch/train/images"
    LABEL_DIR = "data/Pascal VOC 2012.v1-raw.yolov5pytorch/train/labels"
    ds = CustomDataset(
        img_dir = IMG_DIR,
        label_dir = LABEL_DIR,
        transform = transform,
        C = 1000,
        width=448,
        height=448
    )
    
    print(f"img tensor shape {ds[100][0].shape}")
    imgshow(ds[101][0], ds[101][1])
    cv2.waitKey(0)
    '''
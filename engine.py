import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np

from models.loss_function.yololoss import intersection_over_union

def train_one_epoch(model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer,
                    data_loader: DataLoader, device: torch.device, epoch: int, epochs: int):
    
    print(f"Epoch: [{epoch:03d}/{epochs:03d}]")
    mean_loss = []
    
    model.train()
    
    for batch_idx, (batches, targets) in enumerate(data_loader, start=1):
        batches = batches.to(device)
        targets = targets.to(device)
        
        # Feed-Forward
        optimizer.zero_grad()
        outputs = model(batches)
        loss = loss_fn(outputs, targets)
        
        # Back Propagation
        loss.backward()
        optimizer.step()
        
        mean_loss.append(loss.item())
        print(f"\rTraining: {100*batch_idx/len(data_loader):.2f}% Loss: {loss.item():.6f}", end="")

    return sum(mean_loss)/len(mean_loss) 

def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device, num_classes: int, iou_threshold: float):
    model.eval()
    
    n_ground_truths = 0
    boxes_list = [[] for _ in range(0, num_classes)] # (exist_box, confidence, IoU), class 별로 리스트 할당
    
    print()
    for batch_idx, (batches, targets) in enumerate(data_loader, start=1):
        batches = batches.to(device)
        targets = targets.to(device)
        
        outputs = model(batches).reshape(-1, 7, 7, num_classes + 2*5)
        
        # 1st box IoU
        iou_b1 = intersection_over_union(outputs[...,num_classes+1:num_classes+5], targets[...,num_classes+1:num_classes+5])
        # 2nd box IoU
        iou_b2 = intersection_over_union(outputs[...,num_classes+6:num_classes+10], targets[...,num_classes+1:num_classes+5])
        
        # exist box
        # label에서 없는 셀은 FP 처리를 하기 위함
        exist_box = targets[..., num_classes]
        
        # class_pred
        class_pred = torch.argmax(outputs[...,0:num_classes], dim=3)
        
        # confidence (objectness score)
        conf_b1 = outputs[..., num_classes]
        conf_b2 = outputs[..., num_classes+5]
        
        for batch_elements in range(0, outputs.shape[0]):
            for cell_width in range(0, outputs.shape[1]):
                for cell_height in range(0, outputs.shape[2]):
                    exist_ = exist_box[batch_elements, cell_width, cell_height]
                    class_ = class_pred[batch_elements, cell_width, cell_height]
                    
                    conf_b1_ = conf_b1[batch_elements, cell_width, cell_height]
                    conf_b2_ = conf_b2[batch_elements, cell_width, cell_height]
                    
                    iou_b1_ = iou_b1[batch_elements, cell_width, cell_height, 0]
                    iou_b2_ = iou_b2[batch_elements, cell_width, cell_height, 0]
                    
                    boxes_list[class_.item()].append((exist_.item(), conf_b1_.item(), iou_b1_.item()))
                    boxes_list[class_.item()].append((exist_.item(), conf_b2_.item(), iou_b2_.item()))
        print(f"\rTest: {100*batch_idx/len(data_loader):.2f}%", end="")
        
    print()
    class_AP = calculate_AP(boxes_list, iou_threshold)
    
    for i, AP_value in enumerate(class_AP):
        print(f"class: {i}, AP: {AP_value:.4f}")
        
    mAP = calculate_mAP(class_AP)
    print(f"mAP: {mAP:.6f}")
    
def compute_AP(PR_info):
    # recall에 따른 interpolated precision 계산
    # recall 값들 설정
    recall_levels = [x / 10.0 for x in range(11)]  # 0 to 1, 0.1 간격으로
    interpolated_precision = []

    for r in recall_levels:
        # 가장 큰 precision 찾기
        p_max = max([p for p, r_p in PR_info if r_p >= r] or [0.0])
        interpolated_precision.append(p_max)

    # interpolated precision에 대해 recall_levels의 개수로 나누어 AP 계산
    ap = sum(interpolated_precision) / len(recall_levels)
    
    return ap

def calculate_AP(boxes_list, iou_threshold):
    class_AP = []
    
    # per class
    for per_class_boxes in boxes_list:
        per_class_boxes.sort(key=lambda x: x[1], reverse=True) # Confidence 내림차순
        PR_info = []
        
        ground_truths = 1e-6
        TP, FP = 0., 0.
        for values in per_class_boxes:
            if values[0] == 0:
                # box가 존재하지 않는 셀에 대해 예측한 경우 
                FP += 1
            else:
                ground_truths += 1
                if values[2] >= iou_threshold:
                    TP += 1
                else:
                    FP += 1

            precision = TP/(TP+FP)
            recall = TP/ground_truths
            PR_info.append((precision, recall))
        AP = compute_AP(PR_info)
        class_AP.append(AP)
    
    return class_AP

def calculate_mAP(class_AP):
    mAP = np.mean(class_AP)
    return mAP

if __name__ == "__main__":
    pass
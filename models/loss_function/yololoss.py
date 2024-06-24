import torch
from torch import nn

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    '''
    params
        boxes_preds[BATCH_SIZE, 0:4] = 0:4 -> Bouning Boxes Coordinates
    '''
    if box_format == "midpoint":
        box1_x1 = boxes_preds[...,0:1] - boxes_preds[...,2:3] / 2 # x - w/2
        box1_y1 = boxes_preds[...,1:2] - boxes_preds[...,3:4] / 2 # y - h/w
        box1_x2 = boxes_preds[...,0:1] + boxes_preds[...,2:3] / 2 # x + w/2
        box1_y2 = boxes_preds[...,1:2] + boxes_preds[...,3:4] / 2 # y + h/w

        box2_x1 = boxes_labels[...,0:1] - boxes_labels[...,2:3] / 2
        box2_y1 = boxes_labels[...,1:2] - boxes_labels[...,3:4] / 2
        box2_x2 = boxes_labels[...,0:1] + boxes_labels[...,2:3] / 2
        box2_y2 = boxes_labels[...,1:2] + boxes_labels[...,3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[...,0:1]
        box1_y1 = boxes_preds[...,1:2]
        box1_x2 = boxes_preds[...,2:3]
        box1_y2 = boxes_preds[...,3:4] # (N,1)

        box1_x1 = boxes_labels[...,0:1]
        box1_y1 = boxes_labels[...,1:2]
        box1_x2 = boxes_labels[...,2:3]
        box1_y2 = boxes_labels[...,3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area - box2_area - intersection + 1e-6)

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=3):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss()
        # self.mse = mseloss(a,b)
        # def mseloss(a, b):
        #   return torch.mean((a-b)**2)

        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5 #object가 없는 구간에도 loss가 있는데 그 구간이 너무 커서 0.5로 줄임.
        self.lambda_coord = 5 #좌표를 잘 찍느냐, 가중치를 더 주기 위해 5.

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S*(C+B*5)) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(predictions[..., self.C+1:self.C+5], target[...,self.C+1:self.C+5])
        iou_b2 = intersection_over_union(predictions[..., self.C+6:self.C+10], target[...,self.C+1:self.C+5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        #Take the box with highest IoU out of the two prediction
        #Note that bestbox will be indices of 0, 1 for which bbox was best
        # 셀 별로 가장 높은 IoU가 나온 Bounding Box가 무엇인가

        # bestbox.shape.shape = (bz, 7, 7, 1)
        iou_maxes, bestbox = torch.max(ious, dim=0)

        # exists_box.shape = (bz, 7, 7, 1)
        exists_box = target[...,self.C].unsqueeze(3) # in paper this is Iobj_i

        # ======================== #
        #    FOR BOX COORDINATES   #
        # ======================== #

        #Set boxes with no object in them to 0. We only take out one of the two

        # 셀 별로 Target과 IoU가 가장 높은 bbox에 대해 해당 셀에 실제로 Bounding Box가 존재하는지 확인
        box_predictions = exists_box * (
            (
                bestbox*predictions[...,self.C+6:self.C+10] # 2번째 Bounding Box에 있을 거라 예측한 셀들의 bbox 좌표
                + (1-bestbox) * predictions[...,self.C+1:self.C+5] # 1번째 Bounding Box에 있을 거라 예측한 셀들의 bbox 좌표
            ) # (bz, 7, 7, 4), exists_box를 곱하는 이유는 실제 Bounding Box가 있는 셀들만 예측한 Bounding Box에 대해서
            # 비교하기 위함
        )

        box_targets = exists_box * target[...,self.C+1:self.C+5]  # for label

        #Take sqrt of width, height of boxes to ensure that

        # prediction에 대해 sqrt(w), sqrt(h) 구하는 과정
        # torch.sign을 하는 이유는 임시적으로 sqrt 연산을 하기 위해서 양수로 만들어야 하기 때문
        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(
            torch.abs(box_predictions[...,2:4] + 1e-6)
        )
        # target에 대해 sqrt(w), sqrt(h) 구하는 과정
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4]) # for label

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim = -2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ======================== #
        #    FOR OBJECT LOSS       #
        # ======================== #

        #pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[...,self.C+5:self.C+6] + (1-bestbox) * predictions[...,self.C:self.C+1]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),                 # exist box : 0 or 1
            torch.flatten(exists_box * target[...,self.C:self.C+1]),
        )

        # ======================== #
        #    FOR NO OBJECT LOSS    #
        # ======================== #

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[...,self.C:self.C+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[...,self.C:self.C+1], start_dim=1),
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[...,self.C+5:self.C+6], start_dim=1),
            torch.flatten((1 - exists_box) * target[...,self.C:self.C+1], start_dim=1),
        )

        # ======================== #
        #    FOR CLASS LOSS        #
        # ======================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2,),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss # first two rows in paper
            + object_loss # third row in paper
            + self.lambda_noobj * no_object_loss # forth row
            + class_loss # fifth row
        )

        return loss

if __name__ == '__main__':
    loss_fn = YoloLoss()
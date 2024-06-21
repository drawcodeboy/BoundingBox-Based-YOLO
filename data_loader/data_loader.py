from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, boxes = t(img), bboxes

        return img, bboxes

class TennisDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) # Label 파일 위치
        boxes = []
        with open(label_path) as f:
            for label in f.readlines(): # label 내용 불러오기
                # 여기서 x, y는 원래 좌표에 대해 448로 각각 나눈 값이다.
                class_label, x, y, width, height = [ # integer = label, float = x, y, width, height
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0]) # 이미지 파일 위치
        image = Image.open(img_path) # 이미지 불러오기
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

            if label_matrix[i, j, 20] == 0:

                box_coordinates = torch.tensor(
                    # dx, dy, dw, dh
                    [x_cell, y_cell, width, height]
                )

                label_matrix[i, j, 20] = 1
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix

def imgshow(img, box):
    # Target이 있는 index를 2차원에서 1차원으로 flatten하여 나타낸다.
    inx = torch.where(box[..., 20].view(-1) == 1)[0] # 여기가 핵심코드네

    index = []

    for i in inx:
        index.append(divmod(i.numpy(), 7)) # 몇 번째 cell에 있는지 구함

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img.permute(1, 2, 0))

    for _, (i, j) in enumerate(index):
        dx, dy, dw, dh = box[i, j, 21:25]

        x = dx*64 + i*64
        y = dy*64 + j*64
        w = int(dw*448)
        h = int(dh*448)
        print(x, y, w, h)

        xx = np.max((int(x-w/2), 0))
        yy = np.max((int(y-h/2), 0))

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
        )
    
if __name__ == '__main__':
    transform = Compose([transforms.ToTensor(),])
    ds = 
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

yolo_loss = np.load("./saved/loss/yolov1_pascal_loss.npy")
bbb_yolo_loss = np.load("./saved/loss/bb_based_yolov1_pascal_loss.npy")

x = np.arange(1, 100+1, 1)

plt.plot(x, yolo_loss, label='YOLO')
plt.plot(x, bbb_yolo_loss, label='BB-Based YOLO')

plt.legend()
plt.show()
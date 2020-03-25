import cv2
import numpy as np
import random
from os import listdir
from os.path import join

def list_images(directory):
    images = []
    for file in listdir(directory):
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
    return images

path_1 = "D:/DLUT/2020.01.06ICIP/Crop/IV_images/"
#path_2 = "D:/DLUT/2019.09.30 论文/服务器备份/road-scene-infrared-visible-images-master/crop_LR_visible"

img_path_1 = list_images(path_1)
#img_path_2 = list_images(path_2)

for p_1 in img_path_1:
    img_1 = cv2.imread(p_1)
    # img_2 = cv2.imread(p_2)
    width = img_1.shape[0]
    height = img_1.shape[1]
    x = int(width / 8)
    x = int(x * 8)
    y = int(height / 8)
    y = int(y * 8)
    cropped_1 = img_1[0:x, 0:y]  # 裁剪坐标为[y0:y1, x0:x1]
    # cropped_2 = img_2[x:x+256, y:y+256]  # 裁剪坐标为[y0:y1, x0:x1]
    img_gray_1 = cv2.cvtColor(cropped_1, cv2.COLOR_RGB2GRAY)
    #img_gray_2 = cv2.cvtColor(cropped_2, cv2.COLOR_RGB2GRAY)

    cv2.imwrite(p_1,img_gray_1)
    #cv2.imwrite(p_2,img_gray_2)
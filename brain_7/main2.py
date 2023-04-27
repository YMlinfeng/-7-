import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import os

os.chdir(r"D:\pythontry\apython_project\brain_7")

# 读取调整后的图像
img = cv2.imread("adjust1.png")

# # 将图像转换为灰度图像
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 定义滤波核的大小
kernel_size = 7

# 对灰度图像进行中值滤波
median = cv2.medianBlur(img, kernel_size)

# # 将中值滤波结果转换回彩色图像
# result = cv2.cvtColor(median_img, cv2.COLOR_GRAY2BGR)

# # 将所有值为(0,0,0)的像素点的值替换为它们周围像素的平均值或者是中值
# result[np.where((result == [0,0,0]).all(axis=2))] = [255,255,255]

# 显示结果
cv2.imshow('median', median)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
cv2.imwrite('median.png', median)
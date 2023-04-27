import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import os

os.chdir(r"D:\pythontry\apython_project\brain_7")

# 读取图像
img = cv2.imread('median.png')

# 转为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 设定角点检测参数
max_corners = 100
quality_level = 0.3
min_distance = 7
block_size = 7

# 使用goodFeaturesToTrack函数检测角点
corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance, blockSize=block_size)

# 转为整型
corners = corners.astype(int)

# 绘制角点并保存
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

cv2.imwrite('corner_detection.png', img)

# 输出角点数量和位置
print(f"Found {len(corners)} corners:")
for i, corner in enumerate(corners):
    x, y = corner.ravel()
    print(f"Corner {i+1}: ({x}, {y})")

# print(corners)

# print(corners[1][0], corners[1][1])

# 找到 x+y 最大和最小的点
min_point_idx = 0
max_point_idx = 0
for i in range(len(corners)):
    if corners[i][0][0] + corners[i][0][1] < corners[min_point_idx][0][0] + corners[min_point_idx][0][1]:
        min_point_idx = i
    if corners[i][0][0] + corners[i][0][1] > corners[max_point_idx][0][0] + corners[max_point_idx][0][1]:
        max_point_idx = i

min_point = tuple(corners[min_point_idx])
max_point = tuple(corners[max_point_idx])

print("Min point:", min_point)
print("Max point:", max_point)

# 在原图中标记矩形的左上角点和右下角点
img_copy = img.copy()
cv2.circle(img_copy, tuple(min_point[0]), 5, (0, 0, 255), -1)
cv2.circle(img_copy, tuple(max_point[0]), 5, (0, 0, 255), -1)

# 保存结果
cv2.imwrite("result.png", img_copy)

cv2.rectangle(img, tuple(min_point[0]), tuple(max_point[0]), (0, 192, 192), 2)
cv2.imwrite('output.png',img)

img = cv2.imread('target.png')
cv2.rectangle(img, tuple(min_point[0]), tuple(max_point[0]), (0, 192, 192), 2)
cv2.imwrite('final.png',img)


# 显示图片
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import os

os.chdir(r"D:\pythontry\apython_project\brain_7")

# 首先读入待处理的原始图像
img = cv2.imread(r'D:\pythontry\apython_project\brain_7\target.png', cv2.IMREAD_COLOR)

if img is None:
    print("Failed to load image")
else:
    # 获取图像尺寸
    height, width, channels = img.shape

    # 打印图像尺寸信息
    print("Image shape:", height, "x", width, "x", channels)

# 图像预处理：输入图像的尺寸缩放以便于处理
# 首先获取输入文本图像的shape以及对应的最大值
# max_dim = max(img.shape)
# # print(max_dim)
# # 定义size的阈值
# dim_limit = 800
# # 进行size判断
# if max_dim > dim_limit:
#     resize_tor = dim_limit/max_dim
#     img = cv2.resize(img, None, fx=resize_tor, fy=resize_tor)

# 将原始图像拷贝一份
origin_img = img.copy()

# 创建一个空字典用于存储像素值及其出现次数
pixel_dict = {}

# 遍历每个像素并统计出现次数
for y in range(height):
    for x in range(width):
        pixel = tuple(img[y, x])
        if pixel in pixel_dict:
            pixel_dict[pixel] += 1
        else:
            pixel_dict[pixel] = 1

# 计算总像素数
total_pixels = height * width

# 按照比例从大到小对像素字典进行排序
sorted_pixels = sorted(pixel_dict.items(), key=lambda x: x[1]/total_pixels, reverse=True)

# 输出排序后的像素信息
for pixel, count in sorted_pixels:
    proportion = count / total_pixels
    print("Pixel value:", pixel, "Count:", count, "Proportion:", proportion)

num_pixels = len(pixel_dict)
print("Total number of pixels:", num_pixels)

# 取出排名前两种像素点的值
top1_pixel = sorted_pixels[0][0]
top2_pixel = sorted_pixels[1][0]

tar1_pixel = sorted_pixels[2][0]
tar2_pixel = sorted_pixels[3][0]
# tar1_pixel = sorted_pixels[5][0]

# 将排名前两种像素点的值置为白色(255, 255, 255)，其余的值置为黑色(0, 0, 0)
for y in range(height):
    for x in range(width):
        pixel = tuple(img[y, x])
        if pixel == tar1_pixel or pixel == tar2_pixel:
            img[y, x] = [0, 0, 0]
        else:
            img[y, x] = [255, 255, 255]

# 将处理后的图像保存为PNG格式
cv2.imwrite("adjust1.png", img)




# # 重复执行闭操作，从而移除文档图像中的文本
# kernel = np.ones((5,5),np.uint8)
# # 这里以执行次数为4举例，通过与原图对比可以发现，文本图像中的文字已多半被移除。但仍视具体情况需调整迭代次数！
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations= 6)

# # 通过GrabCut算法实现前景对象的分割
# mask = np.zeros(img.shape[:2],np.uint8)
# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
# rect = (20,20,img.shape[1]-20,img.shape[0]-20)
# cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask2[:,:,np.newaxis]

# # 基于canny-edge的边缘检测，处理后的效果如下图所示：
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (11, 11), 0)
# # Edge Detection.
# canny = cv2.Canny(gray, 100, 200)
# canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
# # cv2.imshow('Edge', canny)



# # 对边缘检测后的文档图像数据进行轮廓检测
# con = np.zeros_like(img)
# # 对检测得到的边缘获取其轮廓
# contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# # 保留最大的轮廓，将其绘制在画布上
# page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
# con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)


# # 对轮廓检测后的数据进行角点检测
# con = np.zeros_like(img)
# for c in page:
#     # arcLength计算轮廓周长或曲线长度
#   epsilon = 0.02 * cv2.arcLength(c, True)
#   corners = cv2.approxPolyDP(c, epsilon, True)
#   if len(corners) == 4:
#       break
# cv2.drawContours(con, c, -1, (0, 255, 255), 3)
# cv2.drawContours(con, corners, -1, (0, 255, 0), 10)
# corners = sorted(np.concatenate(corners).tolist())
 
# for index, c in enumerate(corners):
#   character = chr(65 + index)
#   cv2.putText(con, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

# # print(corners[0], corners[1])
# # # 透视变换
# src_points = np.float32([corners[0], corners[1], corners[2], corners[3]])
# print(corners)
# dst_points = np.float32([[0, 800], [0, 0], [600, 800], [600, 0]])
# height, width, channels = origin_img.shape
# # print(height, width)
# # # 计算透视变换矩阵
# M = cv2.getPerspectiveTransform(src_points, dst_points)

# # # 执行透视变换
# oi = origin_img
# # cv2.waitKey(0)
# final = cv2.warpPerspective(origin_img, M, (600, 800))

# cv2.imshow('original', oi)
# cv2.imshow('non_gd', img)
# cv2.imshow('jiaodian', con)
# cv2.imshow('final', final)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



# 开发时间: 2021/10/25 11:42
import pickle
import numpy as np
import cv2
import os
'''
数据预处理文件，用于将数据集按照我规定的方式进行处理，并将处理结果按照测试集和训练集分别保存
'''

# 读取函数，用来读取文件夹中的所有图像，输入参数是文件名
def read_directory(directory_name):
    img_ls=[]
    for filename in os.listdir(directory_name):
        img = cv2.imread(directory_name + "/" + filename)
        img_ls.append(img)
    return img_ls

# 灰度反转函数，将灰度图片的灰度值取补，用于将“黑底白字”图像转化为“白底黑字”
def pic_inv(src):
    img_info = src.shape
    image_height = img_info[0]
    image_weight = img_info[1]
    dst = np.zeros((image_height, image_weight, 1), np.uint8)
    for i in range(image_height):
        for j in range(image_weight):
            grayPixel = src[i][j]
            dst[i][j] = 255 - grayPixel
    return dst

# 判断图像是“白底黑字”还是“黑底白字”
# 只通过周围一圈的像素值进行判断，若白点多则为白底，反之反是
def b_or_w(img):
    area = 0
    height, width = img.shape
    area_all = 2*height + 2*width - 4
    for i in range(height):
        if img[i, 0] == 255:
            area += 1
        if img[i, width-1] == 255:
            area += 1
    for j in range(1, width-1):
        if img[0, j] == 255:
            area += 1
        if img[height-1, j] == 255:
            area += 1
    if area/area_all > 1/2:
        return 'white'
    else:
        return 'black'


# test data processing
img_ls = []
label_ls = []
for i in range(10):         # 在测试数据文件夹下遍历所有图片
    dir = "test_processed/{}".format(i)
    temp = read_directory(dir)
    img_ls += temp
    label_ls += [i]*len(temp)

test_size = len(img_ls)
test_gray = np.zeros((28, 20, test_size))
for i in range(test_size):      # 图像的灰度化与灰度转换，最终尺寸统一并进行归一化
    gray = cv2.cvtColor(img_ls[i], cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if b_or_w(thresh) == 'black':
        gray = pic_inv(gray)
    test_gray[:, :, i] = cv2.resize(gray, (20, 28))
test_gray = test_gray/255
# 标签的one-hot编码
testLabels = np.zeros((10, test_size))
for i in range(test_size):
     testLabels[label_ls[i], i] = 1
# 处理获得的测试数据存储
data_name = 'test_data.pkl'
with open(data_name, 'wb') as f:
    pickle.dump([test_gray, testLabels], f)
print("test data saved to {}".format(data_name))

# # train data processing
# # 步骤完全一致
img_ls = []
label_ls=[]
for i in range(10):
    dir = "train_processed/{}".format(i)
    temp = read_directory(dir)
    img_ls += temp
    label_ls += [i]*len(temp)

train_size = len(img_ls)
train_gray = np.zeros((28, 20, train_size))
for i in range(train_size):
    gray = cv2.cvtColor(img_ls[i], cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if b_or_w(thresh) == 'black':
        gray = pic_inv(gray)
    train_gray[:, :, i] = cv2.resize(gray, (20, 28))
train_gray = train_gray/255

trainLabels = np.zeros((10, train_size))
for i in range(train_size):
     trainLabels[label_ls[i], i] = 1

data_name = 'train_data.pkl'
with open(data_name, 'wb') as f:
    pickle.dump([train_gray, trainLabels], f)
print("train data saved to {}".format(data_name))

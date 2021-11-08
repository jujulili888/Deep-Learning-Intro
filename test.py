# # 开发时间: 2021/10/27 16:00
import pickle
import numpy as np
import matplotlib.pyplot as plt
from building_block import accuracy, fc2
import building_block
'''
用于测试网络1和网络2的模型训练效果
通过打印输出展示测试准确率，通过绘制图像展示预测结果
'''

model_name = 'best_ReLU.pkl'     # 此处读取的是网络1的最佳模型训练效果
f = open(model_name, 'rb')
w, layer_size = pickle.load(f)
L = len(layer_size)
test_name = 'test_data.pkl'     # 读取测试数据
f2 = open(test_name, 'rb')
testData, testLabels = pickle.load(f2)
test_size = testLabels.shape[1]
X_test = testData.reshape((-1, test_size))
# 完成对测试数据的预测
a, z, delta = {}, {}, {}
a[1] = X_test
y = testLabels
for l in range(1, L - 1):
    a[l + 1], z[l + 1] = fc2(w[l], a[l])
z[L] = np.dot(w[L - 1], a[L - 1])
a[L] = building_block.f(z[L])
# 打印测试准确率
print("test accuracy of the best trained model with ReLU+sigmoid: {:.4f}".format(accuracy(a[L], y)))
# 选取四个数字图像，输出其图像及对应的网络预测输出
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.imshow(testData[:, :, 0], cmap='gray')
ax1.set_title('prediction: %d' % (np.argmax(a[L], axis=0))[0])
ax2.imshow(testData[:, :, 5000], cmap='gray')
ax2.set_title('prediction: %d' % (np.argmax(a[L], axis=0))[5000])
ax3.imshow(testData[:, :, 15000], cmap='gray')
ax3.set_title('prediction: %d' % (np.argmax(a[L], axis=0))[15000])
ax4.imshow(testData[:, :, 20000], cmap='gray')
ax4.set_title('prediction: %d' % (np.argmax(a[L], axis=0))[20000])
plt.tight_layout()
plt.savefig("test.png")
plt.close()

import math, pickle
import numpy as np
import matplotlib.pyplot as plt
from building_block import cost, accuracy, fc2, bc2, df
import building_block
'''网络1的训练程序代码'''

if __name__ == '__main__':
    # loading data: loading processed train data and test data
    train_name = 'train_data.pkl'
    f = open(train_name, 'rb')
    trainData, trainLabels = pickle.load(f)
    train_size = trainLabels.shape[1]
    X_train = trainData.reshape((-1, train_size))
    test_name = 'test_data.pkl'
    f2 = open(test_name, 'rb')
    testData, testLabels = pickle.load(f2)
    test_size = testLabels.shape[1]
    X_test = testData.reshape((-1, test_size))

    # Step 2: Network Architecture Design
    # define number of layers
    L = 5
    # define number of neurons in each layer
    layer_size = [560, 512, 256, 64, 10]
    max_epoch = 80      # number of training epoch 200
    mini_batch = 100    # number of sample of each mini batch 100
    beta = 1e-6
    # Step 3: Initializing Network Parameters
    # initialize weights
    w = {}
    for l in range(1, L):
        w[l] = 0.1 * np.random.randn(layer_size[l], layer_size[l-1])

    alpha = 0.001  # initialize learning rate

    # Step 6: Train the Network
    J = []      # array to store cost of each mini batch
    Acc = []    # array to store accuracy of each mini batch
    train_acc = []
    test_acc = []
    for epoch_num in range(max_epoch):
        idxs = np.random.permutation(train_size)
        for k in range(math.ceil(train_size/mini_batch)):
            start_idx = k*mini_batch
            end_idx = min((k+1)*mini_batch, train_size)

            a, z, delta = {}, {}, {}
            batch_indices = idxs[start_idx:end_idx]
            a[1] = X_train[:, batch_indices]
            y = trainLabels[:, batch_indices]
            # forward computation
            for l in range(1, L-1):
                a[l+1], z[l+1] = fc2(w[l], a[l])        # 前向传播时隐藏层使用ReLU激活
            # 最后一层仍采取sigmoid激活
            z[L] = np.dot(w[L-1], a[L-1])
            a[L] = building_block.f(z[L])

            delta[L] = (a[L] - y + beta) * df(z[L])
            # backward computation
            for l in range(L-1, 1, -1):
                delta[l] = bc2(w[l], z[l], delta[l+1], beta)

            # update weights
            for l in range(1, L):
                grad_w = np.dot(delta[l+1], a[l].T)
                w[l] = w[l] - alpha*grad_w

            J.append(cost(a[L], y)/mini_batch)
            Acc.append(accuracy(a[L], y))
        # Step 7: Test the Network
        a[1] = X_test
        y = testLabels
        # forward computation
        for l in range(1, L-1):
            a[l+1], z[l+1] = fc2(w[l], a[l])
        z[L] = np.dot(w[L - 1], a[L - 1])
        a[L] = building_block.f(z[L])
        # 打印每轮的训练损失、训练准确率以及测试准确率
        print('%d/%d  loss: %.4f  train acc: %.4f | test acc: %.4f' %
              (epoch_num+1, max_epoch, J[-1], Acc[-1], accuracy(a[L], y)))
        print("-"*60)
        train_acc.append(Acc[-1])
        test_acc.append(accuracy(a[L], y))
        # 保存最佳模型
        if accuracy(a[L], y) > 0.84:
            model_name = 'best_ReLU.pkl'
            with open(model_name, 'wb') as f:
                pickle.dump([w, layer_size], f)
            print("model saved to {}".format(model_name))
    # 绘制训练损失曲线
    plt.figure()
    plt.plot(J)
    plt.title("cost")
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig("J_test.png")
    plt.close()
    # 绘制训练及测试准确率曲线
    plt.figure()
    plt.plot(train_acc, label='train_acc')
    plt.plot(test_acc, label='test_acc')
    plt.legend()
    plt.title("accuracy on train set and test set")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.savefig("Acc_test.png")
    plt.close()
    # Step 8: Store the Network Parameters：ultimate model
    # save model
    model_name = 'model_ReLU.pkl'
    with open(model_name, 'wb') as f:
        pickle.dump([w, layer_size], f)
    print("model saved to {}".format(model_name))

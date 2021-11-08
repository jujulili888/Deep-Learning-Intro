import math, pickle
import numpy as np
import matplotlib.pyplot as plt
from building_block import cross_entropy, accuracy, fc, bc, softmax

if __name__ == '__main__':
    # Step 1: Data Preparation: loading processed train data and test data
    train_name = 'train_data.pkl'
    f = open(train_name, 'rb')
    trainData, trainLabels = pickle.load(f)
    train_size = trainLabels.shape[1]
    X_train = trainData.reshape((-1,train_size))

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
    max_epoch = 70      # number of training epoch 200
    mini_batch = 1    # number of sample of each mini batch 100
    beta = 1e-4
    # Step 3: Initializing Network Parameters
    # initialize weights
    w, b = {}, {}
    for l in range(1, L):
        w[l] = 0.1 * np.random.randn(layer_size[l], layer_size[l-1])
        b[l] = 0.1 * np.random.randn(layer_size[l], 1)

    alpha = 0.01  # initialize learning rate

    # Step 6: Train the Network
    J = []      # array to store cost of every epoch
    train_acc = []
    test_acc = []
    a_pre = np.zeros((10,train_size))
    gt = np.zeros((10,train_size))
    for epoch_num in range(max_epoch):
        idxs = np.random.permutation(train_size)
        for k in range(math.ceil(train_size/mini_batch)):
            print('\r', k+1, end='', flush=True)        # 训练速度较慢，使用该输出标记进度
            start_idx = k*mini_batch
            end_idx = min((k+1)*mini_batch, train_size)

            a, z, delta = {}, {}, {}
            batch_indices = idxs[start_idx:end_idx]
            a[1] = X_train[:, batch_indices]
            y = trainLabels[:, batch_indices]
            # forward computation
            for l in range(1, L-1):
                a[l+1], z[l+1] = fc(w[l], a[l], b[l])
            # 最后一层采用softmax激活
            z[L] = np.dot(w[L - 1], a[L - 1]) + b[L-1]
            a[L] = softmax(z[L])

            delta[L] = (a[L] - y + beta)
            # backward computation
            for l in range(L-1, 1, -1):
                delta[l] = bc(w[l], z[l], delta[l+1], beta)

            # update weights
            for l in range(1, L):
                grad_w = np.dot(delta[l+1], a[l].T)
                grad_b = delta[l+1]
                w[l] = w[l] - alpha*grad_w
                b[l] = b[l] - alpha*grad_b
            a_pre[:, k] = a[L].flatten()
            gt[:, k] = y.flatten()

        # Step 7: Test the Network
        print()
        a[1] = X_test
        y = testLabels
        # forward computation
        for l in range(1, L-1):
            a[l+1], z[l+1] = fc(w[l], a[l], b[l])

        z[L] = np.dot(w[L - 1], a[L - 1]) + b[L-1]
        a[L] = softmax(z[L])
        # 打印每轮的训练损失、训练准确率以及测试准确率
        print('%d/%d  loss: %.4f  train acc: %.4f | test acc: %.4f' %
              (epoch_num+1, max_epoch, cross_entropy(a_pre, gt)/train_size,
               accuracy(a_pre, gt), accuracy(a[L], y)))
        print("-"*60)
        train_acc.append(accuracy(a_pre, gt))
        test_acc.append(accuracy(a[L], y))
        J.append(cross_entropy(a_pre, gt)/train_size)
        # 保存最佳模型
        if accuracy(a[L], y) > 0.85:
            model_name = 'best_softmax.pkl'
            with open(model_name, 'wb') as f:
                pickle.dump([w, layer_size], f)
            print("model saved to {}".format(model_name))
    # 绘制训练损失曲线
    plt.figure()
    plt.plot(J)
    plt.title("cost")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig("J_softmax.png")
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
    plt.savefig("Acc_softmax.png")
    plt.close()
    # Step 8: Store the Network Parameters: ultimate model
    # save model
    model_name = 'model_softmax.pkl'
    with open(model_name, 'wb') as f:
        pickle.dump([w, layer_size], f)
    print("model saved to {}".format(model_name))

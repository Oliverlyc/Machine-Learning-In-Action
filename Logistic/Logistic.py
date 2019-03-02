import  numpy as np
import matplotlib.pyplot as plt
def load_dataset():
    data_matrix = []
    label_matrix = []
    file = open('./testSet.txt')
    for line in file.readlines():
        line_array = line.strip().split()
        data_matrix.append([1.0, float(line_array[0]), float(line_array[1])])
        label_matrix.append(int(line_array[2]))
    return data_matrix, label_matrix

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

#梯度上升
def grad_ascent(data_matrix_in, class_labels):
    data_matrix = np.mat(data_matrix_in)
    label_matrix = np.mat(class_labels).transpose()#转置
    m, n = data_matrix.shape
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n,1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_matrix - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights

def plot_bestfit(weights):
    data_matrix , label_matrix = load_dataset()
    data_array = np.mat(data_matrix)
    n = data_array.shape[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_matrix[i]) == 1:
            xcord1.append(data_array[i,1])
            ycord1.append(data_array[i,2])
        else:
            xcord2.append(data_array[i,1])
            ycord2.append(data_array[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y.transpose())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stoc_grad_ascent_v1(data_matrix, class_labels):
    m, n = np.array(data_matrix).shape
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(np.sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * np.array(data_matrix)[i]
    return weights
def stoc_grad_ascent_v2(data_matrix, class_labels, iter=300):
    m, n = np.array(data_matrix).shape
    weights = np.ones(n)
    data_index = range(m)
    for j in range(iter):
        for i in range(m):
            alpha = 4/(1.0 + j + i)+0.1
            rand_index = np.random.choice(len(data_index),replace=False)
            h = sigmoid(np.sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * np.array(data_matrix)[rand_index]
    return weights
if __name__ == '__main__':
    x, y = load_dataset()
    weights = stoc_grad_ascent_v2(x, y)
    print(weights)
    plot_bestfit(weights)
import numpy as np
import Logistic
def classify_vector(x, weights):
    prob = Logistic.sigmoid(np.sum(x * weights))
    if prob > 0.5:
        return 1
    else:
        return 0

def colic_test():
    train_file = open('./horseColicTraining.txt')
    test_file = open('./horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in train_file.readlines():
        current_line = line.strip().split('\t')
        line_array = []
        for i in range(21):
            line_array.append(np.float(current_line[i]))
        training_set.append(line_array)
        training_labels.append(np.float(current_line[21]))
    train_weights = Logistic.stoc_grad_ascent_v2(np.array(training_set), training_labels, 500)
    print(train_weights)
    error_count = 0
    number_test_vector = 0.0
    for line in test_file.readlines():
        number_test_vector += 1.0
        current_line = line.strip().split('\t')
        line_array = []
        for i in range(21):
            line_array.append(np.float(current_line[i]))
        if np.int(classify_vector(np.array(line_array), train_weights)) != np.int(current_line[21]):
            error_count += 1
    error_rate = (np.float(error_count) / number_test_vector)
    print("the error rate of this test is : %f" % error_rate)
    return error_rate
def multi_test():
    number_test = 10
    error_sum = 0.0
    for k in range(number_test):
        error_sum += colic_test()
    print("after %d iterations the average error rate is:%f" % (number_test, error_sum/np.float(number_test)))
multi_test()
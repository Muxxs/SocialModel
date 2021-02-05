# coding=utf-8

import tensorflow as tf
import random


def get_data(nums):
    import os
    x_train = []  # 4*1
    y_train = []  # 1*1
    for i in range(nums):
        NewList = []
        for m in list([0, 1, 2, 3]):
            NewList.append(random.random() // 0.1)
        x_train.append(NewList)
    return x_train


x_train = get_data(10)
x_train.append([1, 1, 1, 1])
x_train.append([2, 2, 2, 2])
model = tf.keras.models.load_model('my_model')

r = model.predict(x_train)
for i in range(len(r)):
    print(x_train[i][0] + x_train[i][1] + x_train[i][2] + x_train[i][3],
          int(x_train[i][0] + x_train[i][1] + x_train[i][2] + x_train[i][3] > 20), int(r[i][0] < r[i][1]), r[i][0],
          r[i][1])

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# coding=utf-8

import tensorflow as tf
import datetime
import cv2  # 导入opencv库
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import random


def f(x):
    return x / 255.0


def get_data(nums):
    import os
    x_train = []  # 4*1
    y_train = []  # 1*1
    for i in range(nums):
        NewList = []
        sum = 0
        for m in list([0, 1, 2, 3]):
            num = random.random() // 0.1
            sum += num
            NewList.append(num)
        x_train.append(NewList)
        y_train.append(int(sum > 20))
    return x_train, y_train


def FitModol():
    # Tensorflow model
    x_train, y_train = get_data(300)
    x_test = x_train[:50]
    y_test = y_train[:50]
    y_train = y_train[50:]
    x_train = x_train[50:]

    for i in range(len(x_train)):
        print(x_train[i], y_train[i])

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(4, 1)),
            tf.keras.layers.Dense(2, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)
        ]
    )

    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, batch_size=1)
    model.save('my_model')
    model.evaluate(x_test, y_test, verbose=2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    FitModol()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

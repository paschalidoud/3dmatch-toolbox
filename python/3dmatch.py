#!/usr/bin/env python
"""Train a 3DMatch network
"""
import argparse
import sys

import numpy as np
from keras.layers import Activation, Conv3D, MaxPooling3D, Input, \
                         Flatten
from keras.models import Sequential
from keras import backend as K


def euclidean_distance(D1, D2):
    return K.sqrt(K.square(D1 - D2).sum(axis=1))


def create_network(input_shape):
    model = Sequential([
       Conv3D(64, 3, input_shape=input_shape),
       Activation("relu"),
       Conv3D(64, 3),
       Activation("relu"),
       MaxPooling3D(strides=(2, 2, 2)),
       Conv3D(128, 3),
       Activation("relu"),
       Conv3D(128, 3),
       Activation("relu"),
       Conv3D(256, 3),
       Activation("relu"),
       Conv3D(256, 3),
       Activation("relu"),
       Conv3D(512, 3),
       Activation("relu"),
       Conv3D(512, 3),
       Activation("relu"),
       Conv3D(512, 1),
       Flatten()
    ])

    p1 = Input(shape=(30, 30, 30, 1))
    p2 = Input(shape=(30, 30, 30, 1))
    label = Input(shape=(1,))

    D1 = model(p1)
    D2 = model(p2)

    return model


if __name__ == "__main__":

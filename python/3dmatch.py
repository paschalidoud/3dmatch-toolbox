#!/usr/bin/env python
"""Train 3DMatch
"""
import argparse
import sys

import numpy as np
from keras.layers import Activation, Conv3D, MaxPooling3D, Input
from keras.models import Sequential

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
       Activation("relu")
    ])

    return model


if __name__ == "__main__":
    p1 = Input(shape=(30, 30, 30, 1))
    p2 = Input(shape=(30, 30, 30, 1))
    label = Input(shape=(1,))

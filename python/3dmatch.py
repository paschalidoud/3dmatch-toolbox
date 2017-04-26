#!/usr/bin/env python
"""Train a 3DMatch network
"""
import argparse
import sys

import numpy as np
from keras.layers import Activation, Conv3D, MaxPooling3D, Input, \
                         Flatten, Lambda
from keras.models import Sequential, Model
from keras import backend as K
from keras.optimizers import Adam


def euclidean_distance(D):
    D1 = D[0]
    D2 = D[1]
    return K.sqrt(K.sum(K.square(D1 - D2), axis=1))


def euclidean_distance_output_shape(input_shape):
    return (None, 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
            (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_network(input_shape, weight_file=None):
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

    D1 = model(p1)
    D2 = model(p2)

    distance_layer = Lambda(
        euclidean_distance,
        output_shape=euclidean_distance_output_shape
    )
    distances = distance_layer([D1, D2])

    training_model = Model(inputs=[p1, p2], outputs=distances)

    optimizer = Adam(lr=0.001)
    training_model.compile(
        loss=contrastive_loss,
        optimizer=optimizer
    )

    model.compile(
        loss="mse",
        optimizer=optimizer
    )

    if weight_file:
        training_model.load_weights(weight_file, by_name=True)

    return training_model, model


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a 3DMatch network"
    )

    parser.add_argument(
        "--weight_file",
        help="An initial weights file"
    )
    args = parser.parse_args(argv)

    input_shape = (30, 30, 30, 1)
    p1 = np.random.random((10, input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
    p2 = np.random.random((10, input_shape[0], input_shape[1], input_shape[2], input_shape[3]))

    training_model, model = create_network(
        input_shape,
        weight_file=args.weight_file
    )


if __name__ == "__main__":
    main(sys.argv[1:])

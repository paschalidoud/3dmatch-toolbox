#!/usr/bin/env python
"""Train a 3DMatch network
"""
import argparse
import os
import sys

import numpy as np
from keras.layers import Activation, Conv3D, MaxPooling3D, Input, \
                         Flatten, Lambda
from keras.models import Sequential, Model
from keras import backend as K
from keras.optimizers import Adam


def generate_batches(input_directory, batch_size):
    p1 = [x for x in sorted(os.listdir(input_directory)) if x.endswith(".p1_tdf.bin")]
    p2 = [x for x in sorted(os.listdir(input_directory)) if x.endswith(".p2_tdf.bin")]
    p3 = [x for x in sorted(os.listdir(input_directory)) if x.endswith(".p3_tdf.bin")]

    tdf_grid_dimensions = 30*30*30

    P1 = np.empty((0, tdf_grid_dimensions), dtype=np.float32)
    P2 = np.empty((0, tdf_grid_dimensions), dtype=np.float32)
    labels = []
    # Choose a random index and a random offset to read from the input directory
    for idx in range(batch_size):
        random_idx = np.random.randint(0, len(p1))
        random_offset = np.random.randint(0, 100)
        #print idx,"/", batch_size

        f1 = open(os.path.join(input_directory, p1[random_idx]))
        f1.seek(random_offset * 4 * tdf_grid_dimensions)
        d1 = np.fromfile(f1, count=tdf_grid_dimensions, dtype=np.float32).reshape(-1, tdf_grid_dimensions)

        f2 = open(os.path.join(input_directory, p2[random_idx]))
        f2.seek(random_offset * 4 * tdf_grid_dimensions)
        d2 = np.fromfile(f2, count=tdf_grid_dimensions, dtype=np.float32).reshape(-1, tdf_grid_dimensions)

        f3 = open(os.path.join(input_directory, p3[random_idx]))
        f3.seek(random_offset * 4 * tdf_grid_dimensions)
        d3 = np.fromfile(f3, count=tdf_grid_dimensions, dtype=np.float32).reshape(-1, tdf_grid_dimensions)

        # Add the reference point
        P1 = np.vstack((P1, d1))
        P1 = np.vstack((P1, d1))
        labels.append(1)
        # Add the matching point and the non-matching point
        P2 = np.vstack((P2, d2))
        P2 = np.vstack((P2, d3))
        labels.append(0)

        f1.close()
        f2.close()
        f3.close()

    yield [[P1.reshape((-1, 30, 30, 30, 1)), P2.reshape((-1, 30, 30, 30, 1))], np.array(labels)] 

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
        "training_directory", 
        help="Directory containing the data used for training"
    )
    parser.add_argument(
        "testing_directory", 
        help="Directory containing the data used for testing"
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        help="The batch size to be used"
    )
    parser.add_argument(
        "--weight_file",
        help="An initial weights file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs"
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=5000,
        help="Total number of batches of samples"
    )
    args = parser.parse_args(argv)

    input_shape = (30, 30, 30, 1)
    p1 = np.random.random((10, input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
    p2 = np.random.random((10, input_shape[0], input_shape[1], input_shape[2], input_shape[3]))

    training_model, model = create_network(
        input_shape,
        weight_file=args.weight_file
    )

    #model.predict(p1)
    #training_model.predict([p1, p2])

    training_model.fit_generator(
        generate_batches(args.training_directory, args.batch_size),
        args.steps_per_epoch,
        epochs=args.epochs,
        verbose=2
    )


if __name__ == "__main__":
    main(sys.argv[1:])

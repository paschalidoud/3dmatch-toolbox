#!/usr/bin/env python
"""Train a 3DMatch network
"""
import argparse
from os import path
import sys

import numpy as np
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Activation, Conv3D, MaxPooling3D, Input, \
                         Flatten, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam

import pickle

from utils import BatchProvider


#from theano.compile.nanguardmode import NanGuardMode
class MetricsHistory(Callback):
    def __init__(self, filepath):
        self.fd = open(filepath, "w")
        self.keys = []

    def on_batch_end(self, batch, logs={}):
        if not self.keys:
            self.keys = sorted(logs.keys())
            print >>self.fd, " ".join(self.keys)
        print >>self.fd, " ".join(map(str, [logs[k] for k in self.keys]))
        self.fd.flush()


def collect_test_set(input_directory, n_samples, batch_size=128, voxel_size=0.1,
                     tdf_grid_dims=(30, 30, 30, 1), random_state=0):
    # First set the random state
    prng_state = np.random.get_state()
    np.random.seed(random_state)
    
    # Create the arrays to hold the data
    P1 = np.empty((0,) + tdf_grid_dims, dtype=np.float32)
    P2 = np.empty((0,) + tdf_grid_dims, dtype=np.float32)
    labels = np.empty((0,), dtype=int)

    # Create the generator and collect n_samples from it
    bp = BatchProvider(
        input_directory,
        batch_size,
        voxel_size,
        tdf_grid_dims,
        batches=int(np.ceil(float(n_samples) / batch_size))
    )
    bp.ready()
    for (p1, p2), l in bp:
        P1 = np.vstack([P1, p1])
        P2 = np.vstack([P2, p2])
        labels = np.hstack([labels, l])

        if len(labels) >= n_samples:
            break
    bp.stop()

    # Make our test set but do not return it yet
    test = [[P1[:n_samples], P2[:n_samples]], labels[:n_samples]]

    # We need to reset the random number generator to what it was before
    np.random.set_state(prng_state)

    # Now we 're done
    return test


def euclidean_distance(D):
    D1 = D[0]
    D2 = D[1]
    e = K.epsilon()
    return K.sqrt(K.sum(K.square(D1 - D2), axis=1, keepdims=True) + e)


def euclidean_distance_output_shape(input_shape):
    return (None, 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
            (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def mean_dist(y_true, y_pred):
    return K.mean(y_pred)


def matching_distance(y_true, y_pred):
    return K.mean(y_true * y_pred) / K.mean(y_true)


def non_matching_distance(y_true, y_pred):
    return K.mean((1-y_true) * y_pred) / K.mean(1-y_true)


def create_network(input_shape, weight_file=None, lr=0.001):
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

    optimizer = Adam(lr=lr)
    training_model.compile(
        loss=contrastive_loss,
        optimizer=optimizer,
        metrics=[matching_distance, non_matching_distance, mean_dist]
        #mode=NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=False)
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
        "output_directory",
        help="Save the output files in that directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
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
        default=500,
        help="Total number of batches of samples"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--n_test_samples",
        type=int,
        default=1000,
        help="Number of samples used in the validation set"
    )


    args = parser.parse_args(argv)

    input_shape = (30, 30, 30, 1)

    training_model, model = create_network(
        input_shape,
        weight_file=args.weight_file,
        lr=args.learning_rate
    )

    history = MetricsHistory(path.join(args.output_directory, "metrics.txt"))
    checkpointer = ModelCheckpoint(
        filepath=path.join(args.output_directory, "weights.{epoch:02d}.hdf5"),
        verbose=0
    )
    test_set = collect_test_set(args.testing_directory, args.n_test_samples)
    batch_provider = BatchProvider(args.training_directory, args.batch_size)
    try:
        batch_provider.ready()
        training_model.fit_generator(
            batch_provider,
            args.steps_per_epoch,
            epochs=args.epochs,
            verbose=1,
            validation_data=test_set,
            callbacks=[history, checkpointer]
        )
    except KeyboardInterrupt:
        pass
    batch_provider.stop()


if __name__ == "__main__":
    main(sys.argv[1:])

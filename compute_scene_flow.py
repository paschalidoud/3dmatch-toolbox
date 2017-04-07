#!/usr/env python
"""Script used to compute the scene flow given the 3DMatch desrciptors and
their corresponding keypoints
"""
import argparse
import os
import sys

import numpy as np
from sklearn.neighbors import KDTree


def compute_correspondences(X1, X2, tree_leaf_size=40):
    # Rearrange data so that they have an appropriate format. The first two
    # rows of the descriptors arrays contain the total number of samples  and
    # the corresponing number of features
    n_samples = int(X1[0])
    n_features = int(X1[1])
    print X1, X2

    # Organize the indices of the first input in a tree
    tree = KDTree(
        X1[2:].reshape(n_samples, n_features),
        metric="euclidean",
        leaf_size=tree_leaf_size
    )

    X2_reshaped = X2[2:].reshape(n_samples, n_features)
    indices = {}

    # For all the descriptors from the initial frame compute their
    # corresponding descriptors from the second frame. By corresponding, we
    # mean the descriptor that is closer based on the euclidean distance
    for i in range(5):
        # Compute the distances and indices of the 1th nearest neighbors 
        dist, ind = tree.query(X2_reshaped[i], k=1)
        print i
        print dist
        print ind
        #indices[i] = ind
        print i, ind

    print indices
    return indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Script to comput the scene flow given a set of 3DMatch "
                     "descriptors and their corresponding keypoints")
    )

    parser.add_argument(
        "input_directory",
        help="The directory containing the 3DMatch descriptors"
    )
    parser.add_argument(
        "--start",
        default=0,
        type=int,
        help="Index of the starting frame for the scene flow (inclusive)"
    )
    parser.add_argument(
        "--length",
        default=2,
        type=int,
        help="Length in frames of the computed sceneflow"
    )
    parser.add_argument(
        "--leaf_size",
        default=40,
        type=int,
        help="Leaf size used for the tree used to compute the nearest neighbor"
    )

    args = parser.parse_args(sys.argv[1:])

    input_files = os.listdir(args.input_directory)
    # Find files containing the 3DMatch descriptors and the keypoints
    keypts_files = [x for x in input_files if x.endswith("keypts.bin")]
    desc_files = [x for x in input_files if x.endswith("desc.3dmatch.bin")]

    # Isolate only the files we need according to the provided start and length
    # parameters
    actual_desc_files = desc_files[args.start:args.start+args.length]
    actual_keypts_files = keypts_files[args.start:args.start+args.length]

    print actual_desc_files

    scene_indices = []

    # Iterate through all consecutive frames
    for i, j in zip(actual_desc_files, actual_desc_files[1:]):
        # Read data from the binary files and store them to two numpy array
        input_file = os.path.join(args.input_directory, i)
        f = open(input_file, "r")
        X1 = np.fromfile(f, dtype=np.float32)

        input_file = os.path.join(args.input_directory, j)
        f = open(input_file, "r")
        X2 = np.fromfile(f, dtype=np.float32)

        scene_indices.append(
            compute_correspondences(X1, X2, args.leaf_size)
        )


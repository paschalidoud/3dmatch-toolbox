#!/usr/bin/env python
"""Script used to compute the scene flow given the 3DMatch desrciptors and
their corresponding keypoints
"""
import argparse
import os
import sys

import numpy as np
from sklearn.neighbors import KDTree


def compute_correspondences(D1, D2, tree_leaf_size=40):
    # Organize the indices of the first input in a tree
    tree = KDTree(
        D2,
        metric="euclidean",
        leaf_size=tree_leaf_size
    )

    indices = []

    # For all the descriptors from the initial frame compute their
    # corresponding descriptors from the second frame. By corresponding, we
    # mean the descriptor that is closer based on the euclidean distance
    for i in range(len(D1)):
        # print i, D2[i]
        # Compute the distances and indices of the 1th nearest neighbors
        dist, ind = tree.query(D1[i].reshape(1, -1), k=5)
        print i, dist, ind
        indices.append(ind[0][0])

    return indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Script to comput the scene flow given a set of 3DMatch "
                     "descriptors and their corresponding keypoints")
    )

    parser.add_argument(
        "descriptor_reference_frame",
        help=("The path to the file containing the 3DMatch descriptors for the"
              " first frame in the scene flow")
    )
    parser.add_argument(
        "keypoints_reference_frame",
        help=("The path to the file containing the 3DMatch keypoints for the"
              " first frame in the scene flow")
    )
    parser.add_argument(
        "descriptor_next_frame",
        help=("The path to the file containing the 3DMatch descriptors for the"
              " second frame in the scene flow")
    )
    parser.add_argument(
        "keypoints_next_frame",
        help=("The path to the file containing the 3DMatch keypoints for the"
              " second frame in the scene flow")
    )
    parser.add_argument(
        "output_file",
        help="The path to the file to save the scene flow"
    )

    parser.add_argument(
        "--leaf_size",
        default=40,
        type=int,
        help="Leaf size used for the tree used to compute the nearest neighbor"
    )

    args = parser.parse_args(sys.argv[1:])

    # Load the keypoints for the reference and the next frame
    K1 = np.fromfile(args.keypoints_reference_frame, dtype=np.float32).reshape(-1, 3)
    K2 = np.fromfile(args.keypoints_next_frame, dtype=np.float32).reshape(-1, 3)

    # Load the descriptors for the reference and the next frame
    D1 = np.fromfile(args.descriptor_reference_frame, dtype=np.float32).reshape(-1, 512)
    D2 = np.fromfile(args.descriptor_next_frame, dtype=np.float32).reshape(-1, 512)
    print np.sum(np.isnan(D1)), np.sum(np.isnan(D2))

    if np.sum(np.isnan(D1)) > 0.0:
        print "3DMatch descriptors of first frame contains %d NaN" % np.sum(np.isnan(D1))
        exit(1)
    if np.sum(np.isnan(D2)) > 0:
        print "3DMatch descriptors of next frame contains %d NaN" % np.sum(np.isnan(D2))
        exit(2)

    scene_indices = compute_correspondences(D1, D2, args.leaf_size)
    C = np.hstack([K1, K2[scene_indices]])
        
    with open(args.output_file, "wb") as out:
        C.astype(np.float32).tofile(out)


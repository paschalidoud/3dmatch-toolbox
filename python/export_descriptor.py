#!/usr/bin/env python
"""Script to compute the forward pass of the 3DMatch network in order to
extract each descriptors
"""
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys

import numpy as np

from threedmatch import create_network

from utils import parse_tdf_grid_from_file, extract_point_from_grid


def main(argv):
    parser = argparse.ArgumentParser(
        description="Perform a forward pass of the 3DMatch Network"
    )

    parser.add_argument(
        "pointcloud_path", 
        help="Path containing the pointcloud to be used"
    )
    parser.add_argument(
        "tdf_grid_path",
        help=("Path to the file containing the tdf grid of the pointcloud to be"
              " used")
    )

    parser.add_argument(
        "--weight_file",
        help="An initial weights file"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.1,
        help="Voxel size used to export TDF grid"
    )

    args = parser.parse_args(argv)
    input_shape = (30, 30, 30, 1)
    
    point_idx = 0
    
    origin_ref, tdf_ref = parse_tdf_grid_from_file(args.tdf_grid_path)
    points_ref = np.fromfile(
        args.pointcloud_path,
        dtype=np.float32
    ).reshape(-1, 3)

    p1_tdf = extract_point_from_grid(
        origin_ref,
        tdf_ref,
        points_ref[point_idx],
        args.voxel_size,
        (30, 30, 30, 1)
    ).reshape((1, 30, 30, 30, 1))

    training_model, model = create_network(
        input_shape,
        weight_file=args.weight_file,
        lr=args.learning_rate
    )
    desc = model.predict(p1_tdf)
    print points_ref[0]
    print "desc:",desc


if __name__ == "__main__":
    main(sys.argv[1:])

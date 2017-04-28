#!/usr/bin/env python
"""Script to compute the forward pass of the 3DMatch network in order to
extract each descriptors
"""
import argparse
import os
import sys

import numpy as np

import utils

#from threedmatch import create_network

def main(argv):
    parser = argparse.ArgumentParser(
        description="Perform a forward pass of the 3DMatch Network"
    )

    parser.add_argument(
        "pointcloud_file",
        help="Path to the file containing the pointcloud to be used"
    )
    parser.add_argument(
        "tdf_grid_file",
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
    #input_shape = (1, 30, 30, 30)

    origin_x, origin_y, origin_z, dim_x, dim_y, dim_z, grid = utils.parse_tdf_grid_from_file(
        args.tdf_grid_file
    )

    points = np.fromfile(args.pointcloud_file, dtype=np.float32).reshape(-1, 3)
    points_grid = utils.points_to_grid(
        points,
        origin_x,
        origin_y,
        origin_z,
        args.voxel_size
    )
    
    p = utils.generate_tdf_voxel_grid(points_grid[0], grid, dim_x, dim_y, dim_z)

    #training_model, model = create_network(
    #    input_shape,
    #    weight_file=args.weight_file,
    #    lr=args.learning_rate
    #)
    #desc = model.predict(p)
    print points[0]
    print points_grid[0]
    print "p:", p
    #print "desc:",desc

if __name__ == "__main__":
    main(sys.argv[1:])

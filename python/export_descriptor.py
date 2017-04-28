#!/usr/bin/env python
"""Script to compute the forward pass of the 3DMatch network in order to
extract each descriptors
"""
import argparse
import os
import sys

import numpy as np

from threedmatch import create_network


def parse_tdf_grid_from_file(tdf_grid_file):
   f = open(tdf_grid_file, "rb") 
   origin_x = np.fromfile(f, count=1, dtype=np.float32)
   origin_y = np.fromfile(f, count=1, dtype=np.float32)
   origin_z = np.fromfile(f, count=1, dtype=np.float32)

   dim_x = np.fromfile(f, count=1, dtype=np.int32)
   dim_y = np.fromfile(f, count=1, dtype=np.int32)
   dim_z = np.fromfile(f, count=1, dtype=np.int32)

   pointer = np.fromfile(f, count=1, dtype=np.float32)

   grid = np.fromfile(f, count=dim_x*dim_y*dim_z, dtype=np.float32)

   return origin_x, origin_y, origin_z, dim_x, dim_y, dim_z, grid


def points_to_grid(points, origin_x, origin_y, origin_z, voxel_size):
    origins = np.array([origin_x, origin_y, origin_z]).reshape(1, 3)
    return np.round((points - np.repeat(origins, len(points), axis=0)) / voxel_size)


def generate_tdf_voxel_grid(point, grid, dim_x, dim_y, dim_z, input_shape):
    tdf_voxel_grid = []
     
    z_start = int(point[2] - 15)
    z_stop = int(point[2] + 15)
    y_start = int(point[1] - 15)
    y_stop = int(point[1] + 15)
    x_start = int(point[0] - 15)
    x_stop = int(point[0] + 15)

    for z in range(z_start, z_stop, 1):
        for y in range(y_start, y_stop, 1):
            for x in range(x_start, x_stop, 1):
                tdf_voxel_grid.append(grid[z * dim_x * dim_y + y * dim_x + x])

    return np.array(tdf_voxel_grid).reshape(input_shape)


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

    origin_x, origin_y, origin_z, dim_x, dim_y, dim_z, grid = parse_tdf_grid_from_file(
        args.tdf_grid_file
    )

    points = np.fromfile(args.pointcloud_file, dtype=np.float32).reshape(-1, 3)
    points_grid = points_to_grid(
        points,
        origin_x,
        origin_y,
        origin_z,
        args.voxel_size
    )
    
    p = generate_tdf_voxel_grid(points_grid[0], grid, dim_x, dim_y, dim_z, input_shape).shape

    training_model, model = create_network(
        input_shape,
        weight_file=args.weight_file,
        lr=args.learning_rate
    )
    desc = training_model.predict(p)
    print desc

if __name__ == "__main__":
    main(sys.argv[1:])

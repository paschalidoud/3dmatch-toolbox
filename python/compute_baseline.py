#!/usr/bin/env python
"""Script used to implement the baseline using the 3DMatch descriptor
"""
import argparse
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys

import numpy as np

from threedmatch import create_network
from threedmatch_scene_flow import compute_correspondences, \
                                   compute_projected
from utils import parse_tdf_grid_from_file, extract_point_from_grid


def generate_tdfs(tdf_file, points, voxel_size, tdf_grid_dims, batch_size):
    origin, tdf = parse_tdf_grid_from_file(tdf_file)
    while True:
        batch_tdfs = np.empty((0, 30, 30, 30, 1), dtype=np.float32)
        for i, p in enumerate(points):
            # Compute the tdf voxel grid for each point in the pointcloud
            p_tdf = extract_point_from_grid(
                        origin,
                        tdf,
                        p,
                        voxel_size,
                        tdf_grid_dims
            ).reshape((1, 30, 30, 30, 1))

            # Extract the descriptor for the computed tdf voxel grid
            batch_tdfs = np.vstack((batch_tdfs, p_tdf))
            if batch_tdfs.shape[0] == min(batch_size, len(points)-1):
                # Reset batch_tdfs
                batch_tdfs = np.empty((0, 30, 30, 30, 1), dtype=np.float32)
                yield batch_tdfs

def compute_descriptors(tdf_file, points_file, model, voxel_size=0.1,
                        tdf_grid_dims=(30, 30, 30, 1), batch_size=256.0):
    points = np.fromfile(
        points_file,
        dtype=np.float32
    ).reshape(-1, 3)
    print points.shape

    D = np.empty((0, 512), dtype=np.float32)
    D = np.vstack((
        D,
        model.predict_generator(
            generate_tdfs(
                tdf_file,
                points,
                voxel_size=voxel_size,
                tdf_grid_dims=tdf_grid_dims,
                batch_size=batch_size
            ),
            np.ceil(len(points)/batch_size)

        ))
    )
    print D.shape

    return D, points


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the scene flow using the 3DMatch network as well"
                     " as the difference between the groundtruth sceneflow w.r.t. various"
                     " metrics")
    )
    parser.add_argument(
        "ref_pointcloud_file",
        help="Path to the file containing the pointcloud of the reference frame"
    )
    parser.add_argument(
        "ref_tdf_grid_file",
        help="Path to the file containing the tdf grid of the reference frame"
    )
    parser.add_argument(
        "proj_pointcloud_file",
        help="Path to the file containing the pointcloud of the projected frame"
    )
    parser.add_argument(
        "proj_tdf_grid_file",
        help="Path to the file containing the tdf grid of the projected frame"
    )
    parser.add_argument(
        "groundtruth_file",
        help=("Path to the file containing the groundtruth data for the"
              " reference and the projected frame")
    )
    parser.add_argument(
        "output_directory",
        help="Directory used to store various output files"
    )
    parser.add_argument(
        "frame_id",
        type=int,
        help="Specify the index of the starting pointcloud in the scene flow"
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

    parser.add_argument(
        "--threshold",
        default=10000,
        type=float,
        help="The distance threshold to determine the correspondences"
    )
    parser.add_argument(
        "--k_neighbors",
        default=1,
        type=int,
        help="The distance threshold to determine the correspondences"
    )
    parser.add_argument(
        "--leaf_size",
        default=40,
        type=int,
        help="Leaf size used for the tree used to compute the nearest neighbor"
    )
    parser.add_argument(
        "--search_in_radius",
        action="store_true",
        help="Choose if you wish to search neighbors that are in a radius"
    )
    parser.add_argument(
        "--search_radius",
        default=1.5,
        type=float,
        help="Choose if you wish to search neighbors that are in a radius"
    )

    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Normalize the difference distance with the scene flow norm"
    )
    parser.add_argument(
        "--store_3dmatch_scene_flow",
        action="store_true",
        help=("Set if you wish to save the scene flow computed using the"
              " 3DMatch descriptor")
    )
    parser.add_argument(
        "--store_gt_scene_flow",
        action="store_true",
        help="Set if you wish to save the ground truth scene flow"
    )

    args = parser.parse_args(argv)
    input_shape = (30, 30, 30, 1)

    # Initialize the network to be used for the forward pass
    training_model, model = create_network(
        input_shape,
        weight_file=args.weight_file,
        lr=args.learning_rate
    )

    # Compute the descriptor of the reference and the projected pointclouds
    print "Computing descriptors for the reference pointcloud..."
    D_ref, points_ref = compute_descriptors(
        args.ref_tdf_grid_file,
        args.ref_pointcloud_file,
        model,
        voxel_size=args.voxel_size
    )
    print "Computing descriptors for the projected pointcloud..."
    D_proj, points_proj = compute_descriptors(
        args.proj_tdf_grid_file,
        args.proj_pointcloud_file,
        model,
        voxel_size=args.voxel_size
    )

    # Compute the matches
    print "Compute the correspondences..."
    scene_indices = compute_correspondences(D_ref, D_proj, args.threshold, args.leaf_size)
    print "Keypoints without correspondence %d/%d" % (
        (scene_indices < 0).sum(),
        len(scene_indices)
    )
    print "Compute the matches"
    C = np.hstack(
        [
            points_ref,
            compute_projected(
                points_ref,
                points_proj,
                scene_indices,
                args.k_neighbors,
                args.search_in_radius,
                args.search_radius
            )
         ]
    )
    c_ref = C[:, :3]
    c_proj = C[:, 3:]
    scene_flow_matches = c_proj - c_ref

    # Read the groundtruth data for the specified frame
    gt = np.fromfile(args.groundtruth_file, dtype=np.float32).reshape(-1, 8)
    # Take the points for the starting frame 
    points = gt[:, 0] == args.frame_id
    # Find the points of the reference and the projected point cloud
    d_ref_gt = gt[points, 2:5]
    d_proj_gt = gt[points, 5:]

    # Compute the scene flow
    scene_flow_gt = d_proj_gt - d_ref_gt

    diff = scene_flow_gt - scene_flow_matches
    diff_norm = np.sqrt(np.sum(diff**2, axis=1))
    scen_flow_gt_norm = np.sqrt(np.sum(scene_flow_gt**2, axis=1))

    mean_diff_norm = diff_norm.mean()
    mean_normalized_diff_norm = (diff_norm / scen_flow_gt_norm).mean()

    print "The mean%s euclidean distance between the two scene flows is %f" % (
        " normalized" if args.normalized else "",
        mean_normalized_diff_norm if args.normalized else mean_diff_norm
    )

    # Store the files that should be stored
    if args.store_gt_scene_flow:
        output_file = os.path.join(args.output_directory, "gt_scene_flow.bin")
        with open(output_file, "wb") as out:
            scene_flow_gt.astype(np.float32).tofile(out)

    if args.store_3dmatch_scene_flow:
        output_file = os.path.join(args.output_directory, "3dmatch_scene_flow.bin")
        with open(output_file, "wb") as out:
            C.astype(np.float32).tofile(out)

if __name__ == "__main__":
    main(sys.argv[1:])

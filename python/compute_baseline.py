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
from utils import parse_tdf_grid_from_file, extract_point_from_grid, \
                  save_to_binary_file, filter_data
from google_cloud_utils import append_to_spreadsheet


SPREADSHEET = "1zwaat1QFWDRDVxHtncKBGZPwLBPEMwLPMZXkF-_bCz0"


def generate_tdfs(tdf_file, points, voxel_size, tdf_grid_dims, batch_size):
    origin, tdf = parse_tdf_grid_from_file(tdf_file)

    while True:
        batch_tdfs = np.empty((0, 30, 30, 30, 1), dtype=np.float32)
        for p in points:
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
            if batch_tdfs.shape[0] == batch_size:
                yield batch_tdfs
                # Reset batch_tdfs
                batch_tdfs = np.empty((0, 30, 30, 30, 1), dtype=np.float32)

        if len(batch_tdfs) > 0:
            yield batch_tdfs

        
def compute_descriptors(tdf_file, points_file, model, voxel_size=0.1,
                        tdf_grid_dims=(30, 30, 30, 1), batch_size=256.0):
    points = np.fromfile(
        points_file,
        dtype=np.float32
    ).reshape(-1, 4)[:, :-1]

    D = model.predict_generator(
        generate_tdfs(
            tdf_file,
            points,
            voxel_size=voxel_size,
            tdf_grid_dims=tdf_grid_dims,
            batch_size=batch_size
        ),
        np.ceil(len(points)/batch_size),
        verbose=1
    )
    print D.shape

    return D, points


def set_paths(input_directory, sequence, voxel_size):
    # File to the ground-truth scene flow
    gt_scene_flow = os.path.join(
        input_directory,
        "sf",
        sequence + ".npy",
    )
    # File to the pointcloud at timestamp t
    pointcloud_ref = os.path.join(
        input_directory,
        "velodyne",
        sequence + ".bin"
    )
    # File to the pointcloud at timestamp t+1
    pointcloud_proj = os.path.join(
        input_directory,
        "velodyne_projected",
        sequence + ".bin"
    )
    # File to the tdf voxel grid at timestamp t
    ref_tdf_path = "velodyne_tdf_voxel_size_%s" %(str(voxel_size),)
    tdf_ref = os.path.join(
        input_directory,
        ref_tdf_path,
        sequence + ".voxel_tdf_grid.bin.gz"
    )
    # File to the tdf voxel grid at timestamp t+1
    projected_tdf_path = "velodyne_projected_tdf_voxel_size_%s" %(str(voxel_size),)
    tdf_proj = os.path.join(
        input_directory,
        ref_tdf_path,
        sequence + ".voxel_tdf_grid.bin.gz"
    )
    return gt_scene_flow, pointcloud_ref, pointcloud_proj, tdf_ref, tdf_proj


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the scene flow using the 3DMatch network as well"
                     " as the difference between the groundtruth sceneflow w.r.t. various"
                     " metrics")
    )
    parser.add_argument(
        "input_directory",
        help="Path to the directory containing the pointclouds and the tdf grids"
    )
    parser.add_argument(
        "sequence",
        help="Name of the sequence to be processed"
    )
    parser.add_argument(
        "output_directory",
        help="Directory used to store various output files"
    )
    parser.add_argument(
        "weight_file",
        help="An initial weights file"
    )

    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.1,
        help="Voxel size used to export TDF grid (default:0.1)"
    )

    parser.add_argument(
        "--threshold",
        default=10000,
        type=float,
        help="The distance threshold to determine the correspondences (default:10000)"
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=1,
        help="The distance threshold to determine the correspondences(default:1)"
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
        "--train_from_scratch",
        action="store_true",
        help="Parameter used when training is performed from scratch"
    )
    parser.add_argument(
        "--credentials",
        default=os.path.join(os.path.dirname(__file__), ".credentials"),
        help="The credentials file for the Google API"
    )
    parser.add_argument(
        "--run_in_cluster",
        action="store_true",
        help="Specify whether the script is being executed in the cluster"
    )

    args = parser.parse_args(argv)
    input_shape = (30, 30, 30, 1)

    # Initialize the network to be used for the forward pass
    training_model, model = create_network(
        input_shape,
        weight_file=args.weight_file
    )

    gt_scene_flow_file, ref_pointcloud_file, proj_pointcloud_file, ref_tdf_grid_file, proj_tdf_grid_file =\
        set_paths(
            args.input_directory,
            args.sequence,
            args.voxel_size
        )

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Compute the descriptor of the reference and the projected pointclouds
    print "Computing descriptors for the reference pointcloud..."
    D_ref, points_ref = compute_descriptors(
        ref_tdf_grid_file,
        ref_pointcloud_file,
        model,
        voxel_size=args.voxel_size
    )
    print "Computing descriptors for the projected pointcloud..."
    D_proj, points_proj = compute_descriptors(
        proj_tdf_grid_file,
        proj_pointcloud_file,
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
    print "Compute the  %s-Nearest neighbors matches" %(args.k_neighbors,)
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
    scene_flow_gt = np.load(gt_scene_flow_file)

    diff = scene_flow_gt - scene_flow_matches
    diff_norm = np.sqrt(np.sum(diff**2, axis=1))
    scen_flow_gt_norm = np.sqrt(np.sum(scene_flow_gt**2, axis=1))

    mean_diff_norm = diff_norm.mean()
    mean_normalized_diff_norm = (diff_norm / scen_flow_gt_norm).mean()

    print "The mean%s euclidean distance between the two scene flows is %f" % (
        " normalized" if args.normalized else "",
        mean_normalized_diff_norm if args.normalized else mean_diff_norm
    )
    
    with open(os.path.join(args.output_directory, "%s_statistics.txt" %(args.sequence)), "a") as f:
        f.write("Sequence: %s\n" % args.sequence)
        f.write("k: %f\n" % args.k_neighbors)
        f.write("threshold: %f\n" % args.threshold)
        f.write("Mean euclidean distance: %f\n" % mean_diff_norm)
        f.write("Normalized mean euclidean distance: %f\n" % mean_normalized_diff_norm)

    print "Saving the scene_flow for thr=%.2f" % (args.threshold)
    np.save(
        os.path.join(args.output_directory, args.sequence),
        scene_flow_matches
    )
    # Append results to the spreadsheet
    if not args.run_in_cluster:
        append_to_spreadsheet(
            SPREADSHEET, "Sheet5",
            [[args.sequence, 
              "train_from_scratch" if args.train_from_scratch else "fine_tune",
              args.voxel_size,
              args.k_neighbors,
              args.threshold,
              "%f" % mean_diff_norm]],
            credential_path=args.credentials
        )


if __name__ == "__main__":
    main(sys.argv[1:])

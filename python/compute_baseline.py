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

        
def compute_descriptors(tdf_file, points_file, model, xlim, ylim, zlim, voxel_size=0.1,
                        tdf_grid_dims=(30, 30, 30, 1), batch_size=256.0):
    points = np.fromfile(
        points_file,
        dtype=np.float32
    ).reshape(-1, 3)

    points = filter_data(points, xlim, ylim, zlim)
    print points.shape

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


def set_paths(input_directory, sequence, start_frame):
    end_frame = start_frame + 1
    t = "/".join([input_directory, sequence])
    ref_pointcloud_file = "_".join([t, "%03d" %start_frame, "ref.bin"])
    proj_pointcloud_file = "_".join([t, "%03d" %end_frame, "ref.bin"])

    ref_tdf_grid_file = "_".join([t, "%03d.voxel_tdf_grid.bin.gz" %start_frame])
    proj_tdf_grid_file = "_".join([t, "%03d.voxel_tdf_grid.bin.gz" %end_frame])

    return ref_pointcloud_file, ref_tdf_grid_file, proj_pointcloud_file, proj_tdf_grid_file


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
        "start_frame",
        type=int,
        help="Index of the starting frame"
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
        "--store_descriptors",
        action="store_true",
        help="Set if you wish to save the computed 3DMatch descriptors"
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
    parser.add_argument(
        "--xlim",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-30,30",
        help="The limits of the x-axis"
    )
    parser.add_argument(
        "--ylim",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-30,30",
        help="The limits of the y-axis"
    )
    parser.add_argument(
        "--zlim",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-3,2",
        help="The limits of the y-axis"
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

    args = parser.parse_args(argv)
    input_shape = (30, 30, 30, 1)

    # Initialize the network to be used for the forward pass
    training_model, model = create_network(
        input_shape,
        weight_file=args.weight_file
    )

    ref_pointcloud_file, ref_tdf_grid_file, proj_pointcloud_file, proj_tdf_grid_file = set_paths(
        args.input_directory,
        args.sequence,
        args.start_frame
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
        args.xlim,
        args.ylim,
        args.zlim,
        voxel_size=args.voxel_size
    )
    print "Computing descriptors for the projected pointcloud..."
    D_proj, points_proj = compute_descriptors(
        proj_tdf_grid_file,
        proj_pointcloud_file,
        model,
        args.xlim,
        args.ylim,
        args.zlim,
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
    gt = np.fromfile(args.groundtruth_file, dtype=np.float32).reshape(-1, 8)
    # Take the groundtruth for the frame we are interested in
    gt_start_frame = gt[gt[:, 0] == args.start_frame]
    # Filter points according to the limits
    gt_start_frame = filter_data(gt_start_frame[:, 2:], args.xlim, args.ylim, args.zlim)

    # Find the points of the reference and the projected point cloud
    d_ref_gt = gt_start_frame[:, :3]
    d_proj_gt = gt_start_frame[:, 3:]

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
    
    with open(os.path.join(args.output_directory, "baseline_metrics.txt"), "a") as f:
        f.write("Sequence: %s\n" % args.sequence)
        f.write("Start: %d\n" % args.start_frame)
        f.write("k: %f\n" % args.k_neighbors)
        f.write("threshold: %f\n" % args.threshold)
        f.write("Mean euclidean distance: %f\n" % mean_diff_norm)
        f.write("Normalized mean euclidean distance: %f\n" % mean_normalized_diff_norm)

    # Start saving stuff
    if args.store_descriptors:
        save_to_binary_file(
            os.path.join(
                args.output_directory,
                "x_%d:%d_y_%d:%d_z_%d:%d_3dmatch_descriptors_ref.bin" %(args.xlim[0], args.xlim[1], args.ylim[0], args.ylim[1], args.zlim[0], args.zlim[1])
            ),
            D_ref
        )
        save_to_binary_file(
            os.path.join(
                args.output_directory,
                "x_%d:%d_y_%d:%d_z_%d:%d_3dmatch_descriptors_proj.bin" %(args.xlim[0], args.xlim[1], args.ylim[0], args.ylim[1], args.zlim[0], args.zlim[1])
            ),
            D_proj
        )

    if args.store_gt_scene_flow:
        save_to_binary_file(
            os.path.join(
                args.output_directory,
                "x_%d:%d_y_%d:%d_z_%d:%d_gt_scene_flow.bin" %(args.xlim[0], args.xlim[1], args.ylim[0], args.ylim[1], args.zlim[0], args.zlim[1])
            ),
            scene_flow_gt
        )

    if args.store_3dmatch_scene_flow:
        save_to_binary_file(
            os.path.join(
                args.output_directory,
                "x_%d:%d_y_%d:%d_z_%d:%d_3dmatch_scene_flow_k_%d_thres_%f" %(args.xlim[0], args.xlim[1], args.ylim[0], args.ylim[1], args.zlim[0], args.zlim[1], args.k_neighbors, args.threshold)
            ),
            scene_flow_matches
        )

        save_to_binary_file(
            os.path.join(
                args.output_directory,
                "x_%d:%d_y_%d:%d_z_%d:%d_correspondences_k_%d_thres_%f" %(args.xlim[0], args.xlim[1], args.ylim[0], args.ylim[1], args.zlim[0], args.zlim[1], args.k_neighbors, args.threshold)
            ),
            C
        )
    # Append results to the spreadsheet
    append_to_spreadsheet(
        SPREADSHEET, "Sheet1",
        [[args.sequence, args.start_frame, args.start_frame+1,
          "xlim:[" + str(args.xlim[0]) + "," + str(args.xlim[1]) + "], ylim:[" +
          str(args.ylim[0]) + "," + str(args.ylim[1]) + "], zlim:[" +
          str(args.zlim[0]) + "," + str(args.zlim[1]) + "]",
          "train_from_scratch" if args.train_from_scratch else "fine_tune",
          args.voxel_size, args.k_neighbors, "%f" % mean_diff_norm]],
        credential_path=args.credentials
    )


if __name__ == "__main__":
    main(sys.argv[1:])

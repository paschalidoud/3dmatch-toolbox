#!/usr/bin/env python
"""Script used to generate visualizations for the scene flow
"""
import os
import argparse
import sys

import numpy as np

from generate_points_lines import convert_to_ascii_ply, save_lines, \
                           save_mixed_lines, save_points, save_mixed_points
from utils import filter_data


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate visualization files"
    )

    parser.add_argument(
        "input_directory",
        help="Path to the directory containing the input files"
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
        "--output_directory",
        help="Path to the directory used to save the visualization files"
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
        "--save_as_ply",
        action="store_true",
        help="Select if you want to save the output files also in ply format"
    )
    parser.add_argument(
        "--use_color",
        action="store_true",
        help="Use colors in the ply files"
    )
    parser.add_argument(
        "--rgb_reference",
        nargs="+",
        default=(128, 128, 128),
        help="Color used to to draw points of the reference pointcloud"
    )
    parser.add_argument(
        "--rgb_gt_next",
        nargs="+",
        default=(0, 128, 255),
        help=("Color used to draw points of the second pointcloud according"
              " to the ground truth")
    )
    parser.add_argument(
        "--rgb_3dmatch_next",
        nargs="+",
        default=(255, 0, 0),
        help=("Color used to draw points of the second pointcloud according"
             " to the 3DMatch descriptor")
    )
    parser.add_argument(
        "--draw_lines",
        action="store_true",
        help=("Generate three files, one containing the scene flow of the"
              " groundtruth, one containing the scene flow computed from the"
              " 3dmatch and one containing both scene flows")
    )
    parser.add_argument(
        "--rgb_gt_scene_flow",
        nargs="+",
        default=(128, 128, 128),
        help="The color used to draw the lines in the ground truth scene flow"
    )
    parser.add_argument(
        "--rgb_3dmatch_scene_flow",
        nargs="+",
        default=(255, 0, 0),
        help="The color used to draw the lines in the 3dmatch scene flow"
    )
    parser.add_argument(
        "--draw_points",
        action="store_true",
        help="Save points in a txt format"
    )

    args = parser.parse_args(argv)
    # If output directory is not defined store stuff in the directory
    if args.output_directory is None:
        output_directory = args.input_directory
    else:
        output_directory = args.output_directory

    print "Saving stuff in %s directory" %(output_directory, )

    start_frame = args.start_frame
    end_frame = start_frame + 1

    # Read the groundtruth data for the specified frame
    gt = np.fromfile(args.groundtruth_file, dtype=np.float32).reshape(-1, 8)
    # Take the groundtruth for the frame we are interested in
    gt_start_frame = gt[gt[:, 0] == start_frame]
    # Filter points according to the limits
    gt_start_frame = filter_data(gt_start_frame[:, 2:], args.xlim, args.ylim, args.zlim)

    # Find the points of the reference and the projected point cloud
    d_ref_gt = gt_start_frame[:, :3]
    d_proj_gt = gt_start_frame[:, 3:]

    correspondences_path = os.path.join(
        args.input_directory,
        "correspondences_k_%d_thres_%f" %(args.k_neighbors, args.threshold)
    )
    # Read 3dmatch correspondences from file
    C = np.fromfile(correspondences_path, dtype=np.float32).reshape(-1, 6)
    C_ref = C[:, :3]
    C_proj = C[:, 3:]

    if args.save_as_ply:
        print "Saving data to ply format..."
        # Save all data in ply format
        convert_to_ascii_ply(
            d_ref_gt,
            os.path.join(output_directory, "gt_ref_%d.ply" %(start_frame)),
            args.use_color,
            tuple(args.rgb_reference)
        )

        convert_to_ascii_ply(
            d_proj_gt,
            os.path.join(output_directory, "gt_proj_%d.ply" %(start_frame)),
            args.use_color,
            tuple(args.rgb_gt_next)
        )

        convert_to_ascii_ply(
            C_proj,
            os.path.join(output_directory, "3dmatch_proj_%d.ply" %(start_frame)),
            args.use_color,
            tuple(args.rgb_3dmatch_next)
        )
    
    if args.draw_lines:
        # Save the lines connecting the points in the scene flows
        print "Saving lines..."
        save_lines(
            d_ref_gt,
            d_proj_gt,
            os.path.join(output_directory, "gt_scene_flow_lines_%d_%d.txt" %(start_frame, end_frame)),
            tuple(args.rgb_gt_scene_flow)
        )
        save_lines(
            C_ref,
            C_proj,
            os.path.join(output_directory, "3dmatch_scene_flow_lines_%d_%d.txt" %(start_frame, end_frame)),
            tuple(args.rgb_3dmatch_scene_flow)
        )
        save_mixed_lines(
            d_ref_gt,
            d_proj_gt,
            C_ref,
            C_proj,
            os.path.join(output_directory, "mixed_scene_flow_lines_%d_%d.txt" %(start_frame, end_frame)),
            tuple(args.rgb_gt_scene_flow),
            tuple(args.rgb_3dmatch_scene_flow)
        )

    if args.draw_points:
        print "Saving points..."
        save_points(
            d_ref_gt,
            d_proj_gt,
            os.path.join(output_directory, "gt_scene_flow_points_%d_%d.txt" %(start_frame, end_frame)),
            tuple(args.rgb_reference),
            tuple(args.rgb_gt_next)
        )
        save_points(
            d_ref_gt,
            C_proj,
            os.path.join(output_directory, "3dmatch_scene_flow_points_%d_%d.txt" %(start_frame, end_frame)),
            tuple(args.rgb_reference),
            tuple(args.rgb_3dmatch_next)
        )
        
   
        save_mixed_points(
            d_ref_gt,
            d_proj_gt,
            C_proj,
            os.path.join(output_directory, "mixed_scene_flow_points_%d_%d.txt" %(start_frame, end_frame)),
            tuple(args.rgb_reference),
            tuple(args.rgb_gt_next),
            tuple(args.rgb_3dmatch_next)
        )

if __name__ == "__main__":
    main(sys.argv[1:])

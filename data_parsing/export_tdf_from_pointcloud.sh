#!/bin/bash

declare -a sequences=(
    "gt_2011_09_26_drive_0001_filtered" 
    "gt_2011_09_26_drive_0002_filtered" 
    "gt_2011_09_26_drive_0005_filtered"
    "gt_2011_09_26_drive_0009_filtered"
    "gt_2011_09_26_drive_0011_filtered"
    "gt_2011_09_26_drive_0013_filtered"
    "gt_2011_09_26_drive_0014_filtered"
    "gt_2011_09_26_drive_0015_filtered"
    "gt_2011_09_26_drive_0017_filtered"
    "gt_2011_09_26_drive_0018_filtered"
    "gt_2011_09_26_drive_0019_filtered"
    "gt_2011_09_26_drive_0020_filtered"
    "gt_2011_09_26_drive_0022_filtered"
    "gt_2011_09_26_drive_0023_filtered"
    "gt_2011_09_26_drive_0027_filtered"
    "gt_2011_09_26_drive_0028_filtered"
    "gt_2011_09_26_drive_0029_filtered"
    "gt_2011_09_26_drive_0032_filtered"
    "gt_2011_09_26_drive_0035_filtered"
    "gt_2011_09_26_drive_0039_filtered"
    "gt_2011_09_26_drive_0046_filtered"
    "gt_2011_09_26_drive_0048_filtered"
    "gt_2011_09_26_drive_0051_filtered"
    "gt_2011_09_26_drive_0052_filtered"
    "gt_2011_09_26_drive_0056_filtered"
    "gt_2011_09_26_drive_0057_filtered"
    "gt_2011_09_26_drive_0059_filtered"
    "gt_2011_09_26_drive_0060_filtered"
    "gt_2011_09_26_drive_0061_filtered"
    "gt_2011_09_26_drive_0064_filtered"
    "gt_2011_09_26_drive_0070_filtered"
    "gt_2011_09_26_drive_0079_filtered"
    "gt_2011_09_26_drive_0084_filtered"
    "gt_2011_09_26_drive_0086_filtered"
    "gt_2011_09_26_drive_0087_filtered"
    "gt_2011_09_26_drive_0091_filtered"
    "gt_2011_09_26_drive_0093_filtered"
)


for f in ${sequences[@]}; do
    for i in $(ls ~/Datasets/gt_data/${f} | grep ref.ply); do
        echo ./export_tdf_from_pointcloud ~/Datasets/gt_data/${f}/${i} ~/Datasets/gt_data/${f}/${i:3:-8} 0.1
        ./export_tdf_from_pointcloud ~/Datasets/gt_data/${f}/${i} ~/Datasets/gt_data/${f}/${i:3:-8} 0.1
        gzip ~/Datasets/gt_data/${f}/${i:3:-8}.voxel_tdf_grid.bin
    done
done

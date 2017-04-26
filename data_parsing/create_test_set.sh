#!/bin/bash

declare -a sequences=(
    "gt_2011_09_26_drive_0070_filtered"
    "gt_2011_09_26_drive_0079_filtered"
    "gt_2011_09_26_drive_0084_filtered"
    "gt_2011_09_26_drive_0086_filtered"
    "gt_2011_09_26_drive_0087_filtered"
    "gt_2011_09_26_drive_0091_filtered"
    "gt_2011_09_26_drive_0093_filtered"
)

n_frames=(
    418
    98
    381
    704
    727
    50
    431
)

count=0
for f in ${sequences[@]}; do
    # echo $f
    a=( ${n_frames[$count]} );
    # echo $a
    for ((i=1;i<=a;i++)); do
        # echo $i
        while [ 1 ]; do
            x=${sequences[$RANDOM % ${#sequences[@]} ]}
            if [ "$x" != "$f" ]; then
                break
            fi
        done
        idx=$(( ( $RANDOM % 20 )  + 1 ))
        echo ./parse_point_clouds ~/Datasets/gt_data/$f/${f}_${i}_ref.ply ~/Datasets/gt_data/$f/${f}_${i}_proj.ply ~/Datasets/gt_data/$x/${x}_${idx}_proj.ply ../data/train/ 0.1 100
        ./parse_point_clouds ~/Datasets/gt_data/$f/${f}_${i}_ref.ply ~/Datasets/gt_data/$f/${f}_${i}_proj.ply ~/Datasets/gt_data/$x/${x}_${idx}_proj.ply ~/Desktop/${f} 0.1 100
    done
    count=$((count+1))
done

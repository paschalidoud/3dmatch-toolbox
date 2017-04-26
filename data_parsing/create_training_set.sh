#!/bin/bash

declare -a sequences=(
    #"gt_2011_09_26_drive_0001_filtered" 
    #"gt_2011_09_26_drive_0002_filtered" 
    #"gt_2011_09_26_drive_0005_filtered"
    #"gt_2011_09_26_drive_0009_filtered"
    #"gt_2011_09_26_drive_0011_filtered"
    #"gt_2011_09_26_drive_0013_filtered"
    #"gt_2011_09_26_drive_0014_filtered"
    #"gt_2011_09_26_drive_0015_filtered"
    #"gt_2011_09_26_drive_0017_filtered"
    #"gt_2011_09_26_drive_0018_filtered"
    #"gt_2011_09_26_drive_0019_filtered"
    #"gt_2011_09_26_drive_0020_filtered"
    #"gt_2011_09_26_drive_0022_filtered"
    #"gt_2011_09_26_drive_0023_filtered"
    #"gt_2011_09_26_drive_0027_filtered"
    #"gt_2011_09_26_drive_0028_filtered"
    #"gt_2011_09_26_drive_0029_filtered"
    #"gt_2011_09_26_drive_0032_filtered"
    #"gt_2011_09_26_drive_0035_filtered"
    #"gt_2011_09_26_drive_0039_filtered"
    #"gt_2011_09_26_drive_0046_filtered"
    #"gt_2011_09_26_drive_0048_filtered"
    #"gt_2011_09_26_drive_0051_filtered"
    #"gt_2011_09_26_drive_0052_filtered"
    #"gt_2011_09_26_drive_0056_filtered"
    "gt_2011_09_26_drive_0057_filtered"
    "gt_2011_09_26_drive_0059_filtered"
    "gt_2011_09_26_drive_0060_filtered"
    "gt_2011_09_26_drive_0061_filtered"
    "gt_2011_09_26_drive_0064_filtered"
)

n_frames=(
    #106
    #75
    #152
    #441
    #231
    #142
    #312
    #295
    #112
    #268
    #479
    #84
    #798
    #472
    #186
    #428
    #428
    #388
    #129
    #393
    #123
    #20
    #436
    #76
    #292
    359
    371
    76
    701
    568
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
        echo ./parse_point_clouds ~/Datasets/gt_data/$f/${f}_${i}_ref.ply ~/Datasets/gt_data/$f/${f}_${i}_proj.ply ~/Datasets/gt_data/$x/${x}_${idx}_proj.ply ../data/train/train 0.1 100
        ./parse_point_clouds ~/Datasets/gt_data/$f/${f}_${i}_ref.ply ~/Datasets/gt_data/$f/${f}_${i}_proj.ply ~/Datasets/gt_data/$x/${x}_${idx}_proj.ply ../data/train/train_${f} 0.1 100
    done
    count=$((count+1))
done

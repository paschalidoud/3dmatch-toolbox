#!/bin/bash

sequences=(
    "2011_09_26_drive_0070_filtered"
    "2011_09_26_drive_0079_filtered"
    "2011_09_26_drive_0084_filtered"
    "2011_09_26_drive_0086_filtered"
    "2011_09_26_drive_0087_filtered"
    "2011_09_26_drive_0091_filtered"
    "2011_09_26_drive_0093_filtered"
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
    for i in {0..100}; do
        echo $f
        echo $i
        mkdir -p ~/Desktop/results/${f}/${i}_fine_tuning_weights_49
        mkdir -p ~/Desktop/results/${f}/${i}_from_scratch_weights_49
        echo CUDA_VISIBLE_DEVICES=0 ./compute_baseline.py ~/Datasets/test_set/reference $f $i /ps/geiger/projects/octflownet/ground_truth_filtered/gt_${f}.bin ~/Desktop/results/${f}/${i}_fine_tuning_weights_49 ~/Desktop/3dmatch-training/fine_tuning/voxel_size_0.1-lr_0.0001/weights.49.hdf5 --store_descriptors --store_3dmatch_scene_flow --store_gt_scene_flow --xlim=-10,10 --ylim=-10,10
        CUDA_VISIBLE_DEVICES=0 ./compute_baseline.py ~/Datasets/test_set/reference $f $i /ps/geiger/projects/octflownet/ground_truth_filtered/gt_${f}.bin ~/Desktop/results/${f}/${i}_fine_tuning_weights_49 ~/Desktop/3dmatch-training/fine_tuning/voxel_size_0.1-lr_0.0001/weights.49.hdf5 --store_descriptors --store_3dmatch_scene_flow --store_gt_scene_flow --xlim=-10,10 --ylim=-10,10
        echo CUDA_VISIBLE_DEVICES=0 ./compute_baseline.py ~/Datasets/test_set/reference $f $i /ps/geiger/projects/octflownet/ground_truth_filtered/gt_${f}.bin ~/Desktop/results/${f}/${i}_fine_tuning_weights_49 ~/Desktop/3dmatch-training/fine_tuning/voxel_size_0.1-lr_0.0001/weights.49.hdf5 --store_descriptors --store_3dmatch_scene_flow --store_gt_scene_flow --xlim=-10,10 --ylim=-10,10 --k_neighbors 5
        CUDA_VISIBLE_DEVICES=0 ./compute_baseline.py ~/Datasets/test_set/reference $f $i /ps/geiger/projects/octflownet/ground_truth_filtered/gt_${f}.bin ~/Desktop/results/${f}/${i}_fine_tuning_weights_49 ~/Desktop/3dmatch-training/fine_tuning/voxel_size_0.1-lr_0.0001/weights.49.hdf5 --store_descriptors --store_3dmatch_scene_flow --store_gt_scene_flow --xlim=-10,10 --ylim=-10,10 --k_neighbors 5
        echo CUDA_VISIBLE_DEVICES=0 ./compute_baseline.py ~/Datasets/train_set/reference $f $i /ps/geiger/projects/octflownet/ground_truth_filtered/gt_${f}.bin ~/Desktop/results/${f}/${i}_fine_tuning_weights_49 ~/Desktop/3dmatch-training/fine_tuning/voxel_size_0.1-lr_0.0001/weights.49.hdf5 --store_descriptors --store_3dmatch_scene_flow --store_gt_scene_flow --xlim=-10,10 --ylim=-10,10 --k_neighbors 10
        CUDA_VISIBLE_DEVICES=0 ./compute_baseline.py ~/Datasets/train_set/reference $f $i /ps/geiger/projects/octflownet/ground_truth_filtered/gt_${f}.bin ~/Desktop/results/${f}/${i}_fine_tuning_weights_49 ~/Desktop/3dmatch-training/fine_tuning/voxel_size_0.1-lr_0.0001/weights.49.hdf5 --store_descriptors --store_3dmatch_scene_flow --store_gt_scene_flow --xlim=-10,10 --ylim=-10,10 --k_neighbors 10

        echo CUDA_VISIBLE_DEVICES=0 ./compute_baseline.py ~/Datasets/test_set/reference $f $i /ps/geiger/projects/octflownet/ground_truth_filtered/gt_${f}.bin ~/Desktop/results/${f}/${i}_from_scratch_weights_49 ~/Desktop/3dmatch-training/from_scratch/voxel_size_0.1-lr_0.0001/weights.49.hdf5 --store_descriptors --store_3dmatch_scene_flow --store_gt_scene_flow --xlim=-10,10 --ylim=-10,10
        CUDA_VISIBLE_DEVICES=0 ./compute_baseline.py ~/Datasets/test_set/reference $f $i /ps/geiger/projects/octflownet/ground_truth_filtered/gt_${f}.bin ~/Desktop/results/${f}/${i}_from_scratch_weights_49 ~/Desktop/3dmatch-training/from_scratch/voxel_size_0.1-lr_0.0001/weights.49.hdf5 --store_descriptors --store_3dmatch_scene_flow --store_gt_scene_flow --xlim=-10,10 --ylim=-10,10
        echo CUDA_VISIBLE_DEVICES=0 ./compute_baseline.py ~/Datasets/test_set/reference $f $i /ps/geiger/projects/octflownet/ground_truth_filtered/gt_${f}.bin ~/Desktop/results/${f}/${i}_from_scratch_weights_49 ~/Desktop/3dmatch-training/from_scratch/voxel_size_0.1-lr_0.0001/weights.49.hdf5 --store_descriptors --store_3dmatch_scene_flow --store_gt_scene_flow --xlim=-10,10 --ylim=-10,10 --k_neighbors 5
        CUDA_VISIBLE_DEVICES=0 ./compute_baseline.py ~/Datasets/test_set/reference $f $i /ps/geiger/projects/octflownet/ground_truth_filtered/gt_${f}.bin ~/Desktop/results/${f}/${i}_from_scratch_weights_49 ~/Desktop/3dmatch-training/from_scratch/voxel_size_0.1-lr_0.0001/weights.49.hdf5 --store_descriptors --store_3dmatch_scene_flow --store_gt_scene_flow --xlim=-10,10 --ylim=-10,10 --k_neighbors 5
        echo CUDA_VISIBLE_DEVICES=0 ./compute_baseline.py ~/Datasets/test_set/reference $f $i /ps/geiger/projects/octflownet/ground_truth_filtered/gt_${f}.bin ~/Desktop/results/${f}/${i}_from_scratch_weights_49 ~/Desktop/3dmatch-training/from_scratch/voxel_size_0.1-lr_0.0001/weights.49.hdf5 --store_descriptors --store_3dmatch_scene_flow --store_gt_scene_flow --xlim=-10,10 --ylim=-10,10 --k_neighbors 10
        CUDA_VISIBLE_DEVICES=0 ./compute_baseline.py ~/Datasets/test_set/reference $f $i /ps/geiger/projects/octflownet/ground_truth_filtered/gt_${f}.bin ~/Desktop/results/${f}/${i}_from_scratch_weights_49 ~/Desktop/3dmatch-training/from_scratch/voxel_size_0.1-lr_0.0001/weights.49.hdf5 --store_descriptors --store_3dmatch_scene_flow --store_gt_scene_flow --xlim=-10,10 --ylim=-10,10 --k_neighbors 10
    done
done

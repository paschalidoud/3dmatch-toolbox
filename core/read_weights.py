#!/usr/bin/env python
import h5py
import numpy as np

def read_tensor(f):
    type_id = np.fromfile(f, count=1, dtype=np.uint8)
    print type_id
    
    type_of_size = np.fromfile(f, count=1, dtype=np.uint32)
    print type_of_size
    
    len_name_size = np.fromfile(f, count=1, dtype=np.int32)
    print len_name_size
    
    layer_name = f.read(len_name_size)
    print layer_name
    
    nbdims = np.fromfile(f, count=1, dtype=np.int32)
    print nbdims
    
    weights_shape = np.fromfile(f, count=5, dtype=np.int32)
    print weights_shape
    
    weights = np.fromfile(
        f,
        count=weights_shape[0]*weights_shape[1]*weights_shape[2]*weights_shape[3]*weights_shape[4],
        dtype=np.float32
    )
    if "bias" in layer_name:
        weights = weights.reshape(
            (weights_shape[1])
        )
        
    else:
        weights = weights.reshape(
            (weights_shape[2], weights_shape[3], weights_shape[4], weights_shape[1], weights_shape[0])
        )
    print weights.shape
    print " "
    return layer_name, weights
    

if __name__ == "__main__":
    f = open("3dmatch-weights-snapshot-137000.marvin")
    
    model = {}
    for i in range(18):
        layer_name, weights = read_tensor(f)
        model[layer_name] = weights
    
    f = h5py.File("/tmp/3dmatch-weights-snapshot-137000.marvin", "w")

    # Create inital group
    group1 = f.create_group("model_weights")
    # Create first subgroup
    group21 = f["model_weights"].create_group("input_1")
    group22 = f["model_weights"].create_group("input_2")
    group23 = f["model_weights"].create_group("lambda_1")
    group24 = f["model_weights"].create_group("sequential_1")
    # Create second subgroup
    g = f["model_weights"]
    group31 = g["sequential_1"].create_group("conv3d_1")
    group32 = g["sequential_1"].create_group("conv3d_2")
    group33 = g["sequential_1"].create_group("conv3d_3")
    group34 = g["sequential_1"].create_group("conv3d_4")
    group35 = g["sequential_1"].create_group("conv3d_5")
    group36 = g["sequential_1"].create_group("conv3d_6")
    group37 = g["sequential_1"].create_group("conv3d_7")
    group38 = g["sequential_1"].create_group("conv3d_8")
    group39 = g["sequential_1"].create_group("conv3d_9")

    # Start creating datasets
    s = g["sequential_1"]
    # First convolution layer
    s["conv3d_1"].create_dataset("bias:0", (64,), data=model["conv1.bias"], dtype="f4")
    s["conv3d_1"].create_dataset("kernel:0", (3, 3, 3, 1, 64), data=model["conv1.weight"], dtype="f4")

    # Second convolution layer
    s["conv3d_2"].create_dataset("bias:0", (64,), data=model["conv2.bias"], dtype="f4")
    s["conv3d_2"].create_dataset("kernel:0", (3, 3, 3, 64, 64), data=model["conv2.weight"], dtype="f4")

    # Third convolution layer
    s["conv3d_3"].create_dataset("bias:0", (128,), data=model["conv3.bias"], dtype="f4")
    s["conv3d_3"].create_dataset("kernel:0", (3, 3, 3, 64, 128), data=model["conv3.weight"], dtype="f4")

    # Fourth convolution layer
    s["conv3d_4"].create_dataset("bias:0", (128,), data=model["conv4.bias"], dtype="f4")
    s["conv3d_4"].create_dataset("kernel:0", (3, 3, 3, 128, 128), data=model["conv4.weight"], dtype="f4")

    # Fifth convolution layer
    s["conv3d_5"].create_dataset("bias:0", (256,), data=model["conv5.bias"], dtype="f4")
    s["conv3d_5"].create_dataset("kernel:0", (3, 3, 3, 128, 256), data=model["conv5.weight"], dtype="f4")

    # Sixth convolution layer
    s["conv3d_6"].create_dataset("bias:0", (256,), data=model["conv6.bias"], dtype="f4")
    s["conv3d_6"].create_dataset("kernel:0", (3, 3, 3, 256, 256), data=model["conv6.weight"], dtype="f4")

    # Seventh convolution layer
    s["conv3d_7"].create_dataset("bias:0", (512,), data=model["conv7.bias"], dtype="f4")
    s["conv3d_7"].create_dataset("kernel:0", (3, 3, 3, 256, 512), data=model["conv7.weight"], dtype="f4")

    # Eighth convolution layer
    s["conv3d_8"].create_dataset("bias:0", (512,), data=model["conv8.bias"], dtype="f4")
    s["conv3d_8"].create_dataset("kernel:0", (3, 3, 3, 512, 512), data=model["conv8.weight"], dtype="f4")
    
    # Ninth convolution layer
    s["conv3d_9"].create_dataset("bias:0", (512,), data=model["feat.bias"], dtype="f4")
    s["conv3d_9"].create_dataset("kernel:0", (1, 1, 1, 512, 512), data=model["feat.weight"], dtype="f4")

    f.close()

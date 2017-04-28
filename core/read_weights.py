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
    
    f = h5py.File("/tmp/3dmatch-weights-snapshot-137000.marvin.hdf5")

    g1 = f["model_weights"]
    g2 = g1["sequential_1"]

    conv3d = g2["conv3d_1"]
    data = conv3d["bias:0"]
    data[...] = model["conv1.bias"]
    data = conv3d["kernel:0"]
    data[...] = model["conv1.weight"]

    conv3d = g2["conv3d_2"]
    data = conv3d["bias:0"]
    data[...] = model["conv2.bias"]
    data = conv3d["kernel:0"]
    data[...] = model["conv2.weight"]

    conv3d = g2["conv3d_3"]
    data = conv3d["bias:0"]
    data[...] = model["conv3.bias"]
    data = conv3d["kernel:0"]
    data[...] = model["conv3.weight"]

    conv3d = g2["conv3d_4"]
    data = conv3d["bias:0"]
    data[...] = model["conv4.bias"]
    data = conv3d["kernel:0"]
    data[...] = model["conv4.weight"]

    conv3d = g2["conv3d_5"]
    data = conv3d["bias:0"]
    data[...] = model["conv5.bias"]
    data = conv3d["kernel:0"]
    data[...] = model["conv5.weight"]

    conv3d = g2["conv3d_6"]
    data = conv3d["bias:0"]
    data[...] = model["conv6.bias"]
    data = conv3d["kernel:0"]
    data[...] = model["conv6.weight"]

    conv3d = g2["conv3d_7"]
    data = conv3d["bias:0"]
    data[...] = model["conv7.bias"]
    data = conv3d["kernel:0"]
    data[...] = model["conv7.weight"]

    conv3d = g2["conv3d_8"]
    data = conv3d["bias:0"]
    data[...] = model["conv8.bias"]
    data = conv3d["kernel:0"]
    data[...] = model["conv8.weight"]

    conv3d = g2["conv3d_9"]
    data = conv3d["bias:0"]
    data[...] = model["feat.bias"]
    data = conv3d["kernel:0"]
    data[...] = model["feat.weight"]

    f.close()

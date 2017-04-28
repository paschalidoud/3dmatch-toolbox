#!/usr/bin/env python
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
    weights = weights.reshape(
        (weights_shape[2], weights_shape[3], weights_shape[4], weights_shape[1], weights_shape[0])
    )
    #print weights
    print " "
    return layer_name, weights
    

if __name__ == "__main__":
    f = open("3dmatch-weights-snapshot-137000.marvin")
    
    model = {}
    for i in range(18):
        layer_name, weights = read_tensor(f)
        model[layer_name] = weights

    #print model

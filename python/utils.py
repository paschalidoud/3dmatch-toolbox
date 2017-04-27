import numpy as np

def generate_batch(input_directory, batch_size):
    p1 = [x for x in sorted(os.listdir(input_directory)) if x.endswith(".p1_tdf.bin")]
    p2 = [x for x in sorted(os.listdir(input_directory)) if x.endswith(".p2_tdf.bin")]
    p3 = [x for x in sorted(os.listdir(input_directory)) if x.endswith(".p3_tdf.bin")]

    tdf_grid_dimensions = 30*30*30


    P1 = np.empty((0, tdf_grid_dimensions), dtype=np.float32)
    P2 = np.empty((0, tdf_grid_dimensions), dtype=np.float32)
    labels = []
    # Choose a random index and a random offset to read from the input directory
    for idx in range(batch_size):
        random_idx = np.random.randint(0, len(p1))
        random_offset = np.random.randint(0, 100)
        #print idx,"/", batch_size

        f1 = open(os.path.join(input_directory, p1[random_idx]))
        f1.seek(random_offset * 4 * tdf_grid_dimensions)
        d1 = np.fromfile(f1, count=tdf_grid_dimensions, dtype=np.float32).reshape(-1, tdf_grid_dimensions)
        if np.sum(d1) == 0.0:
            assert "TDF voxel with zeros in %s" %(os.path.join(input_directory, p1[random_idx]),)

        f2 = open(os.path.join(input_directory, p2[random_idx]))
        f2.seek(random_offset * 4 * tdf_grid_dimensions)
        d2 = np.fromfile(f2, count=tdf_grid_dimensions, dtype=np.float32).reshape(-1, tdf_grid_dimensions)
        if np.sum(d2)==0.0:
            assert "TDF voxel with zeros in %s" %(os.path.join(input_directory, p2[random_idx]),)

        f3 = open(os.path.join(input_directory, p3[random_idx]))
        f3.seek(random_offset * 4 * tdf_grid_dimensions)
        d3 = np.fromfile(f3, count=tdf_grid_dimensions, dtype=np.float32).reshape(-1, tdf_grid_dimensions)
        if np.sum(d3)==0.0:
            assert "TDF voxel with zeros in %s" %(os.path.join(input_directory, p3[random_idx]),)

        # Add the reference point
        P1 = np.vstack((P1, d1))
        P1 = np.vstack((P1, d1))
        labels.append(1)
        # Add the matching point and the non-matching point
        P2 = np.vstack((P2, d2))
        P2 = np.vstack((P2, d3))
        labels.append(0)

        f1.close()
        f2.close()
        f3.close()

    return [P1.reshape((-1, 30, 30, 30, 1)), P2.reshape((-1, 30, 30, 30, 1))], np.array(labels)
    #return [P1.reshape((-1, 1, 30, 30, 30)), P2.reshape((-1, 1, 30, 30, 30))], np.array(labels)



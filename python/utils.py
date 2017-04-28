#!/usr/bin/env python
import gzip
import numpy as np

def parse_tdf_grid_from_file(tdf_grid_file):
   f = gzip.open(tdf_grid_file, "rb")
   origin_x = np.fromstring(f.read(4), dtype=np.float32)
   origin_y = np.fromstring(f.read(4), dtype=np.float32)
   origin_z = np.fromstring(f.read(4), dtype=np.float32)

   dim_x = np.fromstring(f.read(4), dtype=np.int32)
   dim_y = np.fromstring(f.read(4), dtype=np.int32)
   dim_z = np.fromstring(f.read(4), dtype=np.int32)

   pointer = np.fromstring(f.read(8), dtype=np.int64)

   grid = np.fromstring(f.read(8*dim_x[0]*dim_y[0]*dim_z[0]), dtype=np.float32)

   f.close()
   return origin_x, origin_y, origin_z, dim_x, dim_y, dim_z, grid


def points_to_grid(points, origin_x, origin_y, origin_z, voxel_size):
    origins = np.array([origin_x, origin_y, origin_z]).reshape(1, 3)
    return np.round((points - np.repeat(origins, len(points), axis=0)) / voxel_size)


def generate_tdf_voxel_grid(point, grid, dim_x, dim_y, dim_z):
    tdf_voxel_grid = []
     
    z_start = int(point[2] - 15)
    z_stop = int(point[2] + 15)
    y_start = int(point[1] - 15)
    y_stop = int(point[1] + 15)
    x_start = int(point[0] - 15)
    x_stop = int(point[0] + 15)

    for z in range(z_start, z_stop, 1):
        for y in range(y_start, y_stop, 1):
            for x in range(x_start, x_stop, 1):
                tdf_voxel_grid.append(grid[z * dim_x * dim_y + y * dim_x + x])

                print grid[z * dim_x * dim_y + y * dim_x + x]
    return np.array(tdf_voxel_grid).reshape(-1, 30, 30, 30, 1)


def convert_points_to_grid(tdf_filename, points_filename, voxel_size=0.1):
    origin_x, origin_y, origin_z, dim_x, dim_y, dim_z, grid = utils.parse_tdf_grid_from_file(
        tdf_filename
    )
    points = np.fromfile(points_filename, dtype=np.float32).reshape(-1, 3)
    points_grid = utils.points_to_grid(
        points,
        origin_x,
        origin_y,
        origin_z,
        voxel_size
    )

    return points_grid

    
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



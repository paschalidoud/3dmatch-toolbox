#!/usr/bin/env python
import gzip
import os
import sys
import threading
import time

import numpy as np

from keras.utils.generic_utils import Progbar


def parse_tdf_grid_from_file(tdf_grid_file):
    with gzip.open(tdf_grid_file, "rb") as f:
        # Change the order of the shape into z, y, x so that they are in the
        # same order as in memory
        origin = np.fromstring(f.read(3*4), dtype=np.float32)
        shape = np.fromstring(f.read(3*4), dtype=np.int32)[::-1]

        # Discard the pointer saved with the struct
        f.read(8)

        # Actually read the data
        grid = np.fromstring(
            #f.read(4 * shape.prod()),
            f.read(),
            dtype=np.float32
        ).reshape(shape)

    return origin, grid


def extract_point_from_grid(origin, grid, point, voxel_size, tdf_grid_dims):
    # Transform the point to grid indexes
    point = np.round((point - origin) / voxel_size).astype(int)

    # Extract the a TDF block surrounding the point
    dims = np.array(tdf_grid_dims[:3]) / 2
    start = point-dims
    end = point+dims

    # Check for off by 1 errors
    under_col = start == -1
    over_col = end >= np.array(grid.shape)[::-1]
    start[under_col] += 1
    end[under_col] += 1
    start[over_col] -= 1
    end[over_col] -= 1


    # NOTE: The data in memory are ordered z, y, x so that x is the fastest
    # changing index
    return np.array(
        grid[start[2]:end[2], start[1]:end[1], start[0]:end[0]]
    ).reshape(tdf_grid_dims)


def generate_tdf_voxel_grid(point_file, tdf_file, point_idx, voxel_size=0.1,
                            tdf_grid_dims=(30, 30, 30, 1)):
    # Read the grid and the points
    origin, grid = parse_tdf_grid_from_file(tdf_file)
    points = np.fromfile(point_file, dtype=np.float32).reshape(-1, 3)

    return extract_point_from_grid(
        origin,
        grid,
        points[point_idx],
        voxel_size,
        tdf_grid_dims
    ).reshape((1,) + tdf_grid_dims)



def _points_in_file(filepath, dtype_size=3*4):
    return os.path.getsize(filepath) / dtype_size


def _random_tuple(max_bs, not_matching=None):
    a = np.random.randint(0, len(max_bs))
    b = np.random.randint(0, max_bs[a])

    if not_matching is not None and (a, b) == not_matching:
        return _random_tuple(max_bs, not_matching)

    return a, b


def generate_batches(input_directory, batch_size, voxel_size=0.1,
                  tdf_grid_dims=(30, 30, 30, 1)):
    reference_tdfs = [
        os.path.join(input_directory, "reference", x)
        for x in sorted(os.listdir(os.path.join(input_directory, "reference")))
        if x.endswith(".gz")
    ]
    projected_tdfs = [
        os.path.join(input_directory, "projected", x)
        for x in sorted(os.listdir(os.path.join(input_directory, "projected")))
        if x.endswith(".gz")
    ]
    reference_points = [
        os.path.join(input_directory, "reference", x)
        for x in sorted(os.listdir(os.path.join(input_directory, "reference")))
        if x.endswith("ref.bin")
    ]
    projected_points = [
        os.path.join(input_directory, "projected", x)
        for x in sorted(os.listdir(os.path.join(input_directory, "projected")))
        if x.endswith("proj.bin")
    ]
    number_of_points = map(_points_in_file, reference_points)

    # Make sure that for every frame there is a reference point cloud and tdf
    # and a projected point cloud and tdf
    assert len(reference_tdfs) == len(reference_points)
    assert len(reference_tdfs) == len(projected_tdfs)
    assert len(reference_tdfs) == len(projected_points)

    while True:
        # These will hold the batch data
        P1 = np.empty((0,) + tdf_grid_dims, dtype=np.float32)
        P2 = np.empty((0,) + tdf_grid_dims, dtype=np.float32)
        labels = []

        for idx in range(batch_size):
            # Choose a random frame and a random point from that frame as the
            # first point
            p1_frame_idx, p1_point_idx = _random_tuple(number_of_points)

            # Choose the second point taking into account if it will be a
            # matching point or not
            matching = np.random.randint(0, 2)
            if matching > 0:
                p2_frame_idx, p2_point_idx = _random_tuple(
                    number_of_points,
                    (p1_frame_idx, p1_point_idx)
                )
            else:
                p2_frame_idx, p2_point_idx = p1_frame_idx, p1_point_idx

            # Load the tdfs for the points
            p1_tdf = generate_tdf_voxel_grid(
                reference_points[p1_frame_idx],
                reference_tdfs[p1_frame_idx],
                p1_point_idx,
                voxel_size,
                tdf_grid_dims
            )
            p2_tdf = generate_tdf_voxel_grid(
                projected_points[p2_frame_idx],
                projected_tdfs[p2_frame_idx],
                p2_point_idx,
                voxel_size,
                tdf_grid_dims
            )

            # Append to the matrices
            P1 = np.vstack([P1, p1_tdf])
            P2 = np.vstack([P2, p2_tdf])
            labels.append(matching)

        yield [[P1, P2], np.array(labels).astype(np.float32)]


class BatchProvider(object):
    def __init__(self, input_directory, batch_size, voxel_size=0.1,
                 tdf_grid_dims=(30, 30, 30, 1), batches=500, verbose=1):
        # This is going to be the amount of cached elements
        N = batch_size * batches

        # Allocate memory for the cached elements
        self.reference = np.empty((N,) + tdf_grid_dims, dtype=np.float32)
        self.projected = np.empty((N,) + tdf_grid_dims, dtype=np.float32)
        self.batch_size = batch_size
        self.batches = batches
        self.voxel_size = voxel_size
        self.tdf_grid_dims = tdf_grid_dims
        self.input_directory = input_directory
        self.verbose = verbose

        # Start a thread to fill the cache
        self.cache_lock = threading.RLock()
        self._producer_thread = threading.Thread(target=self._producer)
        self._producer_thread.daemon = True
        self._producer_thread.start()

        # Member variable to stop the thread (to be set via a call to stop)
        self._stop = False
        self._ready = False

    def ready(self, blocking=True):
        while blocking and not self._ready:
            time.sleep(0.1)
        return self._ready

    def stop(self):
        self._stop = True
        self._producer_thread.join()

    def __iter__(self):
        return self

    def __next__(self):
        return next()

    def next(self):
        N = self.batches * self.batch_size
        idxs1 = np.random.randint(0, N, size=self.batch_size)
        while True:
            idxs2 = np.random.randint(0, N, size=self.batch_size)
            if np.all(idxs1 != idxs2):
                break
        matching = np.random.randint(0, 2, size=self.batch_size)
        proj_idxs = np.select([matching == 1, matching == 0], [idxs1, idxs2])

        with self.cache_lock:
            return [
                [
                    self.reference[idxs1],
                    self.projected[proj_idxs]
                ],
                matching.astype(np.float32)
            ]

    def _producer(self):
        # We will be needing the file lists
        input_directory = self.input_directory
        reference_tdfs = [
            os.path.join(input_directory, "reference", x)
            for x in sorted(os.listdir(os.path.join(input_directory, "reference")))
            if x.endswith(".gz")
        ]
        projected_tdfs = [
            os.path.join(input_directory, "projected", x)
            for x in sorted(os.listdir(os.path.join(input_directory, "projected")))
            if x.endswith(".gz")
        ]
        reference_points = [
            os.path.join(input_directory, "reference", x)
            for x in sorted(os.listdir(os.path.join(input_directory, "reference")))
            if x.endswith("ref.bin")
        ]
        projected_points = [
            os.path.join(input_directory, "projected", x)
            for x in sorted(os.listdir(os.path.join(input_directory, "projected")))
            if x.endswith("proj.bin")
        ]
        number_of_points = map(_points_in_file, reference_points)

        # Make sure that for every frame there is a reference point cloud and tdf
        # and a projected point cloud and tdf
        assert len(reference_tdfs) == len(reference_points)
        assert len(reference_tdfs) == len(projected_tdfs)
        assert len(reference_tdfs) == len(projected_points)

        # Keep note of the passes so that we can be sure to unlock the cache
        # for reading and and also show progress during the first pass
        passes = 0
        if self.verbose > 0:
            prog = Progbar(self.batches)

        while True:
            # Acquire the lock for the whole first pass
            if passes == 0:
                self.cache_lock.acquire()

            for batch_idx in range(self.batches):
                # We 're done stop now
                if self._stop:
                    return

                # Pick a frame in random
                frame = np.random.randint(0, len(reference_points))

                # Load the tdf grids and the point clouds
                origin_ref, tdf_ref = parse_tdf_grid_from_file(reference_tdfs[frame])
                points_ref = np.fromfile(
                    reference_points[frame],
                    dtype=np.float32
                ).reshape(-1, 3)
                origin_proj, tdf_proj = parse_tdf_grid_from_file(projected_tdfs[frame])
                points_proj = np.fromfile(
                    projected_points[frame],
                    dtype=np.float32
                ).reshape(-1, 3)

                # Pick random points to generate tdf blocks
                idxs = np.random.randint(
                    0,
                    number_of_points[frame],
                    size=self.batch_size
                )

                # Do the copy to the cache but make sure you lock first and unlock
                # afterwards
                with self.cache_lock:
                    for i, idx in enumerate(idxs):
                        try:
                            self.reference[batch_idx*self.batch_size + i] = \
                                extract_point_from_grid(
                                    origin_ref,
                                    tdf_ref,
                                    points_ref[idx],
                                    self.voxel_size,
                                    self.tdf_grid_dims
                                )
                            self.projected[batch_idx*self.batch_size + i] = \
                                extract_point_from_grid(
                                    origin_proj,
                                    tdf_proj,
                                    points_proj[idx],
                                    self.voxel_size,
                                    self.tdf_grid_dims
                                )
                        except Exception as e:
                            sys.stderr.write("Exception caught in producer thread\n")
                            sys.stderr.write("Frame: %s\n" % (reference_points[frame],))
                            sys.stderr.write("Point: %d\n" % (idx,))
                            sys.stderr.write(str(e))

                # Show progress if it is the first pass
                if passes == 0 and self.verbose > 0:
                    prog.update(batch_idx + 1)

            # Release the lock if it was the first pass
            if passes == 0:
                self._ready = True
                self.cache_lock.release()

            # Count the passes
            passes += 1


def save_to_binary_file(output_file, data, dtype=np.float32):
    with open(output_file, "wb") as out:
        data.astype(dtype).tofile(out)

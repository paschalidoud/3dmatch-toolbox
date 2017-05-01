#!/usr/bin/env python
""" Script to convert data from the KITI dataset to the corresponding ply
format
"""
import os

import numpy as np


def create_header_file(f, vertex_size, use_color):
    # Add the header
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    # Add the size of the existing vertices
    f.write("element vertex " + str(vertex_size) + "\n")
    f.write("property float32 x\n")
    f.write("property float32 y\n")
    f.write("property float32 z\n")
    if use_color:
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
    f.write("end_header\n")


def parse_bin_data_to_ascii_ply(
    P,
    output_filename,
    use_color,
    rgb
):
    # Write data to the output file
    f = open(output_filename, "wb")
    create_header_file(f, P.shape[0], use_color)

    for x in P:
        if use_color:
            f.write(" ".join(map(str, [x[0], x[1], x[2], rgb[0], rgb[1], rgb[2]])))
        else:
            f.write(" ".join(map(str, [x[0], x[1], x[2]])))
        f.write("\n")

    f.close()

def save_lines(P1, P2, output_filename, rgb):
    with open(output_filename, "w") as f:
        f.write("%d\n" %len(P1))
        f.write("9\n")

        for x in np.hstack((P1, P2)):
            f.write(" ".join(
                map(str, [x[0], x[1], x[2], x[3], x[4], x[5], rgb[0], rgb[1], rgb[2]]))
            )
            f.write("\n")
    f.close()

def save_mixed_lines(P1, P2, P3, P4, output_filename, rgb1, rgb2):
    n_points = len(P1) + len(P3)
    with open(output_filename, "w") as f:
        f.write("%d\n" %n_points)
        f.write("9\n")

        for x in np.hstack((P1, P2)):
            f.write(" ".join(
                map(str, [x[0], x[1], x[2], x[3], x[4], x[5], rgb1[0], rgb1[1], rgb1[2]]))
            )
            f.write("\n")

        for x in np.hstack((P3, P4)):
            f.write(" ".join(
                map(str, [x[0], x[1], x[2], x[3], x[4], x[5], rgb2[0], rgb2[1], rgb2[2]]))
            )
            f.write("\n")

    f.close()

def save_points(P1, P2, output_filename, rgb1, rgb2):
    n_points = len(P1) + len(P2)
    with open(output_filename, "w") as f:
        f.write("%d\n" %n_points)
        f.write("6\n")
        for x in P1:
            f.write(" ".join(
                map(str, [x[0], x[1], x[2], rgb1[0], rgb1[1], rgb1[2]]))
            )
            f.write("\n")

        for x in P2:
            f.write(" ".join(
                map(str, [x[0], x[1], x[2], rgb2[0], rgb2[1], rgb2[2]]))
            )
            f.write("\n")
    f.close()


def save_mixed_points(P1, P2, P3, output_filename, rgb1, rgb2, rgb3):
    n_points = len(P1) + len(P2) + len(P3)
    with open(output_filename, "w") as f:
        f.write("%d\n" %n_points)
        f.write("6\n")

        for x in P1:
            f.write(" ".join(
                map(str, [x[0], x[1], x[2], rgb1[0], rgb1[1], rgb1[2]]))
            )
            f.write("\n")
        for x in P2:
            f.write(" ".join(
                map(str, [x[0], x[1], x[2], rgb2[0], rgb2[1], rgb2[2]]))
            )
            f.write("\n")
        for x in P3:
            f.write(" ".join(
                map(str, [x[0], x[1], x[2], rgb3[0], rgb3[1], rgb3[2]]))
            )
            f.write("\n")
    f.close()

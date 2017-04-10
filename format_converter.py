#!/usr/bin/env python
""" Script to convert data from the KITI dataset to the corresponding ply
format
"""
import argparse
import os
import sys

import numpy as np


def create_header_file(f, vertex_size):
    # Add the header
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    # Add the size of the existing vertices
    f.write("element vertex " + str(vertex_size) + "\n")
    f.write("property float32 x\n")
    f.write("property float32 y\n")
    f.write("property float32 z\n")
    f.write("end_header\n")

def parse_ascii_data_to_ascii_ply(input_filename, output_filename):
    # Read data from file
    with open(input_filename, "rb") as f:
        l = f.readlines()
    data = [x.strip().split() for x in l]

    # Write data to the output file
    f = open(output_filename, "wb")
    create_header_file(f, len(data))

    # Start adding the elements
    for idx, x in enumerate(data):
        # Each x contains 4 values, the first 3 values are the x, y, z points
        # and the 4th value is the reflectance point. We only need the x, y, z
        # values
        f.write(" ".join(x[:3]))
        f.write("\n")

    f.close()

def parse_bin_data_to_ascii_ply(input_filename, output_filename):
    # Read data from binary file and store them to a numpy array
    f = open(input_filename, "r")
    data = np.fromfile(f, dtype=np.float32)
    # Write data to the output file
    f = open(output_filename, "wb")
    create_header_file(f, data.shape[0]/4)

    for idx in range(0, len(data), 4):
        #print idx, "/", len(data)
        #print " ".join(map(str, [data[idx], data[idx+1], data[idx+2]]))
        f.write(" ".join(map(str, [data[idx], data[idx+1], data[idx+2]])))
        f.write("\n")

    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to convert KITI data to ply format"
    )

    parser.add_argument(
        "input_directory",
        help="The directory containing the files to be transformed in ply format"
    )
    parser.add_argument(
        "output_directory",
        help="The directory where transformed data should be stored"
    )

    args = parser.parse_args(sys.argv[1:])

    input_filenames = os.listdir(args.input_directory)
    # Change directory for everything to work
    os.chdir(args.input_directory)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    processed = 0 

    for input_file in input_filenames:
        # Create a output filename
        output_file = os.path.join(args.output_directory, input_file)[:-3] + "ply"

        # Check if the input files are binary in order to call the appropriate parser
        if input_file.endswith("bin"):
            parse_bin_data_to_ascii_ply(input_file, output_file)
        else:
            # Replace the extension of the output file
            parse_ascii_data_to_ascii_ply(input_file, output_file)

        processed += 1
        print("Processed %d out of %d" %(processed, len(input_filenames)))

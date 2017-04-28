#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include "marvin.hpp"

#define CUDA_NUM_THREADS 512
#define CUDA_MAX_NUM_BLOCKS 2880

#define IS_PLY_BINARY false

struct tdf_struct {
    float origin_x;
    float origin_y;
    float origin_z;
    int dim_x;
    int dim_y;
    int dim_z;
    float * tdf_values;
};

int random_number(int max_value, int min_value) {
    return rand() % max_value + min_value;
}

// CUDA kernel function to compute TDF voxel grid values given a point cloud (warning: approximate, but fast)
__global__
void ComputeTDF(int CUDA_LOOP_IDX, float * voxel_grid_occ, float * voxel_grid_TDF,
                int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
                float voxel_size, float trunc_margin) {

  int voxel_idx = CUDA_LOOP_IDX * CUDA_NUM_THREADS * CUDA_MAX_NUM_BLOCKS + blockIdx.x * CUDA_NUM_THREADS + threadIdx.x;
  if (voxel_idx > (voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z))
    return;

  int pt_grid_z = (int)floor((float)voxel_idx / ((float)voxel_grid_dim_x * (float)voxel_grid_dim_y));
  int pt_grid_y = (int)floor(((float)voxel_idx - ((float)pt_grid_z * (float)voxel_grid_dim_x * (float)voxel_grid_dim_y)) / (float)voxel_grid_dim_x);
  int pt_grid_x = (int)((float)voxel_idx - ((float)pt_grid_z * (float)voxel_grid_dim_x * (float)voxel_grid_dim_y) - ((float)pt_grid_y * (float)voxel_grid_dim_x));

  int search_radius = (int)round(trunc_margin / voxel_size);

  if (voxel_grid_occ[voxel_idx] > 0) {
    voxel_grid_TDF[voxel_idx] = 1.0f; // on surface
    return;
  }

  // Find closest surface point
  for (int iix = max(0, pt_grid_x - search_radius); iix < min(voxel_grid_dim_x, pt_grid_x + search_radius + 1); ++iix)
    for (int iiy = max(0, pt_grid_y - search_radius); iiy < min(voxel_grid_dim_y, pt_grid_y + search_radius + 1); ++iiy)
      for (int iiz = max(0, pt_grid_z - search_radius); iiz < min(voxel_grid_dim_z, pt_grid_z + search_radius + 1); ++iiz) {
        int iidx = iiz * voxel_grid_dim_x * voxel_grid_dim_y + iiy * voxel_grid_dim_x + iix;
        if (voxel_grid_occ[iidx] > 0) {
          float xd = (float)(pt_grid_x - iix);
          float yd = (float)(pt_grid_y - iiy);
          float zd = (float)(pt_grid_z - iiz);
          float dist = sqrtf(xd * xd + yd * yd + zd * zd) / (float)search_radius;
          if ((1.0f - dist) > voxel_grid_TDF[voxel_idx])
            voxel_grid_TDF[voxel_idx] = 1.0f - dist;
        }
      }
}

tdf_struct compute_tdf_grid(
    float truncated_margin,
    float voxel_size,
    int voxel_grid_padding,
    float * points,
    int num_pts
) {
  // Compute the minimum and maximum value from the points
  float voxel_grid_origin_x = points[0]; 
  float voxel_grid_origin_y = points[1]; 
  float voxel_grid_origin_z = points[2]; 
  float voxel_grid_max_x = points[0];
  float voxel_grid_max_y = points[1];
  float voxel_grid_max_z = points[2];
  for (int pt_idx = 0; pt_idx < num_pts; ++pt_idx) {
    voxel_grid_origin_x = min(voxel_grid_origin_x, points[pt_idx * 3 + 0]);
    voxel_grid_origin_y = min(voxel_grid_origin_y, points[pt_idx * 3 + 1]);
    voxel_grid_origin_z = min(voxel_grid_origin_z, points[pt_idx * 3 + 2]);
    voxel_grid_max_x = max(voxel_grid_max_x, points[pt_idx * 3 + 0]);
    voxel_grid_max_y = max(voxel_grid_max_y, points[pt_idx * 3 + 1]);
    voxel_grid_max_z = max(voxel_grid_max_z, points[pt_idx * 3 + 2]);
  }
  std::cout << "Initial voxel_grid_origin_x: " << voxel_grid_origin_x << std::endl;
  std::cout << "Initial voxel_grid_origin_y: " << voxel_grid_origin_y << std::endl;
  std::cout << "Initial voxel_grid_origin_z: " << voxel_grid_origin_z << std::endl;

  // Create a occupancy grid according to the maximum and minimum values of the point cloud
  int voxel_grid_dim_x = round((voxel_grid_max_x - voxel_grid_origin_x) / voxel_size) + 1 + voxel_grid_padding * 2;
  int voxel_grid_dim_y = round((voxel_grid_max_y - voxel_grid_origin_y) / voxel_size) + 1 + voxel_grid_padding * 2;
  int voxel_grid_dim_z = round((voxel_grid_max_z - voxel_grid_origin_z) / voxel_size) + 1 + voxel_grid_padding * 2;
  
  // Compute the minimum value (m) in each dimension after adding the voxel_grid_padding 
  voxel_grid_origin_x = voxel_grid_origin_x - voxel_grid_padding * voxel_size + voxel_size / 2;
  voxel_grid_origin_y = voxel_grid_origin_y - voxel_grid_padding * voxel_size + voxel_size / 2;
  voxel_grid_origin_z = voxel_grid_origin_z - voxel_grid_padding * voxel_size + voxel_size / 2;

  std::cout << "voxel_grid_origin_x: " << voxel_grid_origin_x << std::endl;
  std::cout << "voxel_grid_origin_y: " << voxel_grid_origin_y << std::endl;
  std::cout << "voxel_grid_origin_z: " << voxel_grid_origin_z << std::endl;

  std::cout << "Size of TDF voxel grid: " << voxel_grid_dim_x << 
               " x " << voxel_grid_dim_y << " x " << voxel_grid_dim_z << std::endl;
  std::cout << "Computing TDF voxel grid..." << std::endl;

  // Convert data to a voxel occupancy grid
  float * voxel_grid_occ = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  // Initialize occupancy grid with 0s
  memset(voxel_grid_occ, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);
  for (int pt_idx = 0; pt_idx < num_pts; ++pt_idx) {
    // Transform each point from meter to "voxel coordinates"
    int pt_grid_x = round((points[pt_idx * 3 + 0] - voxel_grid_origin_x) / voxel_size);
    int pt_grid_y = round((points[pt_idx * 3 + 1] - voxel_grid_origin_y) / voxel_size);
    int pt_grid_z = round((points[pt_idx * 3 + 2] - voxel_grid_origin_z) / voxel_size);
    // For each point in the point cloud assign it to a voxel in the occupancy
    // grid and set this voxel to be equal to one
    int v_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
    voxel_grid_occ[v_idx] = 1.0f;
  }

  // Initialize TDF voxel grid
  float * voxel_grid_TDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  memset(voxel_grid_TDF, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);

  // Copy voxel grids to GPU memory
  float * gpu_voxel_grid_occ;
  float * gpu_voxel_grid_TDF;
  cudaMalloc(&gpu_voxel_grid_occ, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
  cudaMalloc(&gpu_voxel_grid_TDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
  marvin::checkCUDA(__LINE__, cudaGetLastError());
  cudaMemcpy(
    gpu_voxel_grid_occ,
    voxel_grid_occ,
    voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float),
    cudaMemcpyHostToDevice
  );
  cudaMemcpy(
    gpu_voxel_grid_TDF,
    voxel_grid_TDF,
    voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float),
    cudaMemcpyHostToDevice
  );
  marvin::checkCUDA(__LINE__, cudaGetLastError());

  int CUDA_NUM_LOOPS = (int)ceil((float)(voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z) / (float)(CUDA_NUM_THREADS * CUDA_MAX_NUM_BLOCKS));

  for (int CUDA_LOOP_IDX = 0; CUDA_LOOP_IDX < CUDA_NUM_LOOPS; ++CUDA_LOOP_IDX) {
    ComputeTDF <<< CUDA_MAX_NUM_BLOCKS, CUDA_NUM_THREADS >>>(CUDA_LOOP_IDX, gpu_voxel_grid_occ, gpu_voxel_grid_TDF,
        voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
        voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
        voxel_size, truncated_margin);
  }

  // Load TDF voxel grid from GPU to CPU memory
  cudaMemcpy(
    voxel_grid_TDF,
    gpu_voxel_grid_TDF,
    voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float),
    cudaMemcpyDeviceToHost
  );
  marvin::checkCUDA(__LINE__, cudaGetLastError());

  tdf_struct tdf;
  tdf.origin_x = voxel_grid_origin_x;
  tdf.origin_y = voxel_grid_origin_y;
  tdf.origin_z = voxel_grid_origin_z;
  tdf.dim_x = voxel_grid_dim_x;
  tdf.dim_y = voxel_grid_dim_y;
  tdf.dim_z = voxel_grid_dim_z;
  tdf.tdf_values = voxel_grid_TDF;

  return tdf;
}

int main(int argc, char *argv[]) {

  // Check if the command line arguments are correct
  if (argc != 4) {
    std::cout << "Usage: Generate 30x30x30 3D patch for each point in the pointcloud" << std::endl;
    std::cout << "input_pointcloud: Input file containing the reference pointcloud to be processed" << std::endl;
    std::cout << "output_prefix: Output prefix of the files used to store the computed descriptors and keypoints" << std::endl;
    std::cout << "voxel_size: Voxel size of the local 3D path " << std::endl;
    return(1);
  }

  std::string input_pointcloud_filename(argv[1]);
  std::string out_prefix_filename(argv[2]);
  float voxel_size = std::stof(argv[3]);
  int voxel_grid_padding = 15;
  float truncated_margin = voxel_size * 5;


  std::ifstream input_pointcloud_file(input_pointcloud_filename.c_str());
  if (!input_pointcloud_file) {
    std::cerr << "Point cloud file not found." << std::endl;
    return -1;
  }
  int num_pts = 0;
  for (int line_idx = 0; line_idx < 7; ++line_idx) {
    std::string line_str;
    std::getline(input_pointcloud_file, line_str);
    if (line_idx == 2) {
      std::istringstream tmp_line(line_str);
      std::string tmp_line_prefix;
      tmp_line >> tmp_line_prefix;
      tmp_line >> tmp_line_prefix;
      tmp_line >> num_pts;
    }
  }
  if (num_pts == 0) {
    std::cerr << "Third line of .ply file does not tell me number of points." << std::endl;
    return 0;
  }

  float * points = new float[num_pts * 3]; // Nx3 matrix saved as float array (row-major order)
  if (IS_PLY_BINARY) {
    std::cout << "Reading point cloud in binary format..." << std::endl;
    input_pointcloud_file.read((char*)points, sizeof(float) * num_pts * 3);
  }
  else {
    std::cout << "Reading point cloud in ascii format..." << std::endl;
    // This is to read ply files that are in ascii format
    float ptx, pty, ptz;
    int i = 0;
    while (input_pointcloud_file >> ptx >> pty >> ptz) {
      points[i + 0] = ptx;
      points[i + 1] = pty;
      points[i + 2] = ptz;
      // std::cout << "ptx: " << ptx << " pty: " << pty << " ptz: " << ptz << std::endl;
      i += 3;
    }
  }
  input_pointcloud_file.close();
  std::cout << "Loaded input point cloud with " << num_pts << " points!" << std::endl;

  tdf_struct reference_tdf = compute_tdf_grid(
    truncated_margin,
    voxel_size,
    voxel_grid_padding,
    points,
    num_pts
  );

  // Save keypoints as binary file (Nx30x30 float array, row-major order)
  std::string out_saveto_path = out_prefix_filename + ".voxel_tdf_grid.bin";
  std::ofstream out_file(out_saveto_path, std::ios::binary);

  out_file.write((char*)&reference_tdf, sizeof(reference_tdf));
  out_file.write((char*)reference_tdf.tdf_values, sizeof(float)*reference_tdf.dim_x*reference_tdf.dim_y*reference_tdf.dim_z);
  out_file.close();
  std::cout << "TDF voxel grid saved" << std::endl;
}

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#define CUDA_NUM_THREADS 512
#define CUDA_MAX_NUM_BLOCKS 2880

#define IS_PLY_BINARY false
//#define min(a, b) ((a) < (b) ? (a) : (b))
//#define max(a, b) ((a) > (b) ? (a) : (b))

int main(int argc, char *argv[]) {

  // Check if the command line arguments are correct
  if (argc != 5) {
    std::cout << "Usage: Generate 30x30x30 3D patch for each point in the pointcloud" << std::endl;
    std::cout << "reference_pointcloud: Input file containing the reference pointcloud to be processed" << std::endl;
    std::cout << "corresponding_pointcloud: Input file containing the corresponding pointcloud to be processed" << std::endl;
    std::cout << "output_prefix: Output prefix of the files used to store the computed descriptors and keypoints" << std::endl;
    std::cout << "voxel_size: Voxel size of the local 3D path " << std::endl;
    return(1);
  }

  std::string reference_pointcloud_filename(argv[1]);
  std::string corresponding_pointcloud_filename(argv[2]);
  std::string out_prefix_filename(argv[3]);
  float voxel_size = std::stof(argv[4]);
  int voxel_grid_padding = 15;

  std::ifstream reference_pointcloud_file(reference_pointcloud_filename.c_str());
  if (!reference_pointcloud_file) {
    std::cerr << "Point cloud file not found." << std::endl;
    return -1;
  }
  int num_pts = 0;
  for (int line_idx = 0; line_idx < 7; ++line_idx) {
    std::string line_str;
    std::getline(reference_pointcloud_file, line_str);
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
  
  float * reference_points = new float[num_pts * 3]; // Nx3 matrix saved as float array (row-major order)
  if (IS_PLY_BINARY) {
    std::cout << "Reading point cloud in binary format..." << std::endl;
    reference_pointcloud_file.read((char*)reference_points, sizeof(float) * num_pts * 3);
  }
  else {
    std::cout << "Reading point cloud in ascii format..." << std::endl;
    // This is to read ply files that are in ascii format
    float ptx, pty, ptz;
    int i = 0;
    while (reference_pointcloud_file >> ptx >> pty >> ptz) {
      reference_points[i + 0] = ptx;
      reference_points[i + 1] = pty;
      reference_points[i + 2] = ptz;
      // std::cout << "ptx: " << ptx << " pty: " << pty << " ptz: " << ptz << std::endl;
      i += 3;
    }
  }
  reference_pointcloud_file.close();
  std::cout << "Loaded reference point cloud with " << num_pts_corresponding << " points!" << std::endl;

  std::ifstream corresponding_pointcloud_file(corresponding_pointcloud_filename.c_str());
  if (!corresponding_pointcloud_file) {
    std::cerr << "Point cloud file not found." << std::endl;
    return -1;
  }
  int num_pts_corresponding = 0;
  for (int line_idx = 0; line_idx < 7; ++line_idx) {
    std::string line_str;
    std::getline(corresponding_pointcloud_file, line_str);
    if (line_idx == 2) {
      std::istringstream tmp_line(line_str);
      std::string tmp_line_prefix;
      tmp_line >> tmp_line_prefix;
      tmp_line >> tmp_line_prefix;
      tmp_line >> num_pts_corresponding;
    }
  }
  if (num_pts_corresponding == 0) {
    std::cerr << "Third line of .ply file does not tell me number of points." << std::endl;
    return 0;
  }
  
  float * corresponding_points = new float[num_pts_corresponding * 3]; // Nx3 matrix saved as float array (row-major order)
  if (IS_PLY_BINARY) {
    std::cout << "Reading corresponding point cloud in binary format..." << std::endl;
    corresponding_pointcloud_file.read((char*)corresponding_points, sizeof(float) * num_pts * 3);
  }
  else {
    std::cout << "Reading corresponding point cloud in ascii format..." << std::endl;
    // This is to read ply files that are in ascii format
    float ptx, pty, ptz;
    int i = 0;
    while (corresponding_pointcloud_file >> ptx >> pty >> ptz) {
      corresponding_points[i + 0] = ptx;
      corresponding_points[i + 1] = pty;
      corresponding_points[i + 2] = ptz;
      // std::cout << "ptx: " << ptx << " pty: " << pty << " ptz: " << ptz << std::endl;
      i += 3;
    }
  }
  // This is to read ply files that are in binary format
  //pointcloud_file.read((char*)pts, sizeof(float) * num_pts * 3);
  corresponding_pointcloud_file.close();
  std::cout << "Loaded reference point cloud with " << num_pts_corresponding << " points!" << std::endl;

  float trunc_margin = voxel_size * 5;

  // Compute point cloud coordinates of the origin voxel (0,0,0) of the voxel grid
  float voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z;
  float voxel_grid_max_x, voxel_grid_max_y, voxel_grid_max_z;
  voxel_grid_origin_x = reference_points[0]; voxel_grid_max_x = reference_points[0];
  voxel_grid_origin_y = reference_points[1]; voxel_grid_max_y = reference_points[1];
  voxel_grid_origin_z = reference_points[2]; voxel_grid_max_z = reference_points[2];
  for (int pt_idx = 0; pt_idx < num_pts; ++pt_idx) {
    voxel_grid_origin_x = min(voxel_grid_origin_x, pts[pt_idx * 3 + 0]);
    voxel_grid_origin_y = min(voxel_grid_origin_y, pts[pt_idx * 3 + 1]);
    voxel_grid_origin_z = min(voxel_grid_origin_z, pts[pt_idx * 3 + 2]);
    voxel_grid_max_x = max(voxel_grid_max_x, pts[pt_idx * 3 + 0]);
    voxel_grid_max_y = max(voxel_grid_max_y, pts[pt_idx * 3 + 1]);
    voxel_grid_max_z = max(voxel_grid_max_z, pts[pt_idx * 3 + 2]);
  }

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

  std::cout << "Size of TDF voxel grid: " << voxel_grid_dim_x << " x " << voxel_grid_dim_y << " x " << voxel_grid_dim_z << std::endl;
  std::cout << "Computing TDF voxel grid..." << std::endl;

  // Compute surface occupancy grid
  float * voxel_grid_occ = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  // Initialize occupancy grid with 0s
  memset(voxel_grid_occ, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);
  for (int pt_idx = 0; pt_idx < num_pts; ++pt_idx) {
    // Transform each point from meter to "voxel coordinates"
    int pt_grid_x = round((pts[pt_idx * 3 + 0] - voxel_grid_origin_x) / voxel_size);
    int pt_grid_y = round((pts[pt_idx * 3 + 1] - voxel_grid_origin_y) / voxel_size);
    int pt_grid_z = round((pts[pt_idx * 3 + 2] - voxel_grid_origin_z) / voxel_size);
    // For each point in the point cloud assign it to a voxel in the occupancy grid and set this voxel to be equal to one 
    voxel_grid_occ[pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x] = 1.0f;
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
  cudaMemcpy(gpu_voxel_grid_occ, voxel_grid_occ, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_voxel_grid_TDF, voxel_grid_TDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
  marvin::checkCUDA(__LINE__, cudaGetLastError());

  int CUDA_NUM_LOOPS = (int)ceil((float)(voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z) / (float)(CUDA_NUM_THREADS * CUDA_MAX_NUM_BLOCKS));

  for (int CUDA_LOOP_IDX = 0; CUDA_LOOP_IDX < CUDA_NUM_LOOPS; ++CUDA_LOOP_IDX) {
    ComputeTDF <<< CUDA_MAX_NUM_BLOCKS, CUDA_NUM_THREADS >>>(CUDA_LOOP_IDX, gpu_voxel_grid_occ, gpu_voxel_grid_TDF,
        voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
        voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
        voxel_size, trunc_margin);
  }

  // Load TDF voxel grid from GPU to CPU memory
  cudaMemcpy(voxel_grid_TDF, gpu_voxel_grid_TDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
  marvin::checkCUDA(__LINE__, cudaGetLastError());
}

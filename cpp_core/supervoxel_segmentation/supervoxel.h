#include "codelibrary/base/log.h"
#include "codelibrary/geometry/io/xyz_io.h"
#include <vector>

void WritePoints(const char* filename,
                 int n_supervoxels,
                 const cl::Array<cl::RPoint3D>& points,
                 const cl::Array<int>& labels);


std::vector<int> computeSupervoxel(std::string input_file, int k_neighbors, 
                        double resolution, std::string save_file);

#include <string>

bool tile_point_clouds(std::string firstPointCloud="", 
					   std::string secondPointCloud="",
					   int maxPointsPerTile = 1000000, 
					   int minPointsPerTile= 100,
					   bool voxelGridFlag = false,
					   float voxelGridFilterSize= 0.05, 
					   float overlapTiles = 0.0,
					   int projectionDirection = -1,
					   bool verbose = false);


bool resave_point_cloud(std::string firstPointCloud="", 
					   std::string secondPointCloud="",
					   bool verbose = false);
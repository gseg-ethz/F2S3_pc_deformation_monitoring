#include <stdio.h>
#include <vector>
#include <numeric>
#include <chrono>
#include <cmath>
#include <pcl/io/ply_io.h>
#include <omp.h>
#include <iostream>
#include <sstream>
//#include <boost/program_options.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/console/print.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
#include <assert.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <boost/filesystem.hpp>
#include "pc_tiling.h"

#define EPS 1e-9
//namespace po = boost::program_options;



float median(std::vector<float> &v)
{
	size_t n = v.size() / 2;
	nth_element(v.begin(), v.begin() + n, v.end());
	return v[n];
}

float median_point_cloud_resolution(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	// Initialize the kdtree and fit to the point cloud
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);

	std::vector<int> pointIdxNKNSearch(2);
	std::vector<float> pointNKNSquaredDistance(2);
	std::vector<float> pointDistances;

	for (int i = 0; i < cloud->size(); i++) 
	{
		kdtree.nearestKSearch(cloud->points[i], 2, pointIdxNKNSearch, pointNKNSquaredDistance);
		pointDistances.push_back(pointNKNSquaredDistance[1]);
	}

	return std::sqrt(median(pointDistances));
}

void best_fit_plane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::ModelCoefficients::Ptr coefficients, float medianResolution)
{
	// Create inlier point cloud
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	// Mandatory
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(medianResolution);

	seg.setInputCloud(cloud);
	seg.segment(*inliers, *coefficients);

}


void overlap_bounding_boxes(pcl::PointXYZ minPtCloud1, pcl::PointXYZ maxPtCloud1,
							pcl::PointXYZ minPtCloud2, pcl::PointXYZ maxPtCloud2,
							pcl::PointXYZ &overlapMin, pcl::PointXYZ &overlapMax,
							std::vector<float> &overlapArea)
{ 
	/*Computes the overlap of the two bounding boxes together with the area of the sides.*/

	// Check that mins are actually smaller than max
	assert(minPtCloud1.x < maxPtCloud1.x);
	assert(minPtCloud1.y < maxPtCloud1.y);
	assert(minPtCloud1.z < maxPtCloud1.z);
	assert(minPtCloud2.x < maxPtCloud2.x);
	assert(minPtCloud2.y < maxPtCloud2.y);
	assert(minPtCloud2.z < maxPtCloud2.z);

	// Compute border values of the bounding box overlap
	overlapMin.x = std::max(minPtCloud1.x, minPtCloud2.x);
	overlapMin.y = std::max(minPtCloud1.y, minPtCloud2.y);
	overlapMin.z = std::max(minPtCloud1.z, minPtCloud2.z);

	overlapMax.x = std::min(maxPtCloud1.x, maxPtCloud2.x);
	overlapMax.y = std::min(maxPtCloud1.y, maxPtCloud2.y);
	overlapMax.z = std::min(maxPtCloud1.z, maxPtCloud2.z);

	// Compute the area of each side of the overlap
	overlapArea.push_back((overlapMax.y - overlapMin.y) * (overlapMax.z - overlapMin.z));
	overlapArea.push_back((overlapMax.x - overlapMin.x) * (overlapMax.z - overlapMin.z));
	overlapArea.push_back((overlapMax.x - overlapMin.x) * (overlapMax.y - overlapMin.y));

}

void filter_based_on_bb(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn,
						pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut, 
						pcl::PointXYZ overlapMin, 
						pcl::PointXYZ overlapMax)
{
	pcl::CropBox<pcl::PointXYZ> boxFilter;
	boxFilter.setMin(Eigen::Vector4f(overlapMin.x, overlapMin.y, overlapMin.z, 1.0));
	boxFilter.setMax(Eigen::Vector4f(overlapMax.x, overlapMax.y, overlapMax.z, 1.0));
	boxFilter.setInputCloud(cloudIn);
	boxFilter.filter(*cloudOut);
}



pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_grid_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn, float leafSize)
{ 
	/* Performes the voxel grid filtering of the point clouds. If the span of the bounding box is to large in comparison to the size of the voxels
	   the point clout is split into parts, which are filtered and then merged back into one point cloud. 

	*/

	// If the voxel is empty just return the point cloud back
	if (cloudIn->size() == 0)
	{
		return cloudIn;
	}

	// Compute the extent (bounding box) of the point cloud
 	pcl::PointXYZ bbMin, bbMax;
	pcl::getMinMax3D(*cloudIn, bbMin, bbMax);


	// Compute the nubmer of voxels needed
	float side_x = bbMax.x - bbMin.x;
	float side_y = bbMax.y - bbMin.y;
	float side_z = bbMax.z - bbMin.z;

	// Test if the size of the point cloud relative to the size of the voxels would cause overflow
	int64_t numberOfVoxels= (side_x* side_y * side_z) / (leafSize * leafSize * leafSize);
	

	// If the size of the point cloud is too big -> split point cloud into 8 voxels and performe the computation 
	if (numberOfVoxels > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
	{
		pcl::PointXYZ minPT_1 = bbMin; pcl::PointXYZ minPT_2 = bbMin; pcl::PointXYZ minPT_3 = bbMin; pcl::PointXYZ minPT_4 = bbMin;
		pcl::PointXYZ minPT_5 = bbMin; pcl::PointXYZ minPT_6 = bbMin; pcl::PointXYZ minPT_7 = bbMin; pcl::PointXYZ minPT_8 = bbMin;

		pcl::PointXYZ maxPT_1 = bbMax; pcl::PointXYZ maxPT_2 = bbMax; pcl::PointXYZ maxPT_3 = bbMax; pcl::PointXYZ maxPT_4 = bbMax;
		pcl::PointXYZ maxPT_5 = bbMax; pcl::PointXYZ maxPT_6 = bbMax; pcl::PointXYZ maxPT_7 = bbMax; pcl::PointXYZ maxPT_8 = bbMax;

		maxPT_1.x = minPT_1.x + side_x / 2 + EPS; maxPT_1.y = minPT_1.y + side_y / 2 + EPS; maxPT_1.z = minPT_1.z + side_z / 2 + EPS;

		minPT_2.x = minPT_2.x + side_x / 2 - EPS; 
		maxPT_2.x = minPT_2.x + side_x / 2 + EPS; maxPT_2.y = minPT_2.y + side_y / 2 + EPS; maxPT_2.z = minPT_2.z + side_z / 2 + EPS;

		minPT_3.z = minPT_3.z + side_z / 2 - EPS;
		maxPT_3.x = minPT_3.x + side_x / 2 + EPS; maxPT_3.y = minPT_3.y + side_y / 2 + EPS; maxPT_3.z = minPT_3.z + side_z / 2 + EPS;

		minPT_4.x = minPT_4.x + side_x / 2 - EPS; minPT_4.z = minPT_4.z + side_z / 2 - EPS;
		maxPT_4.x = minPT_4.x + side_x / 2 + EPS; maxPT_4.y = minPT_4.y + side_y / 2 + EPS; maxPT_4.z = minPT_4.z + side_z / 2 + EPS;

		minPT_5.y = minPT_5.y + side_y / 2 - EPS;
		maxPT_5.x = minPT_5.x + side_x / 2 + EPS; maxPT_5.y = minPT_5.y + side_y / 2 + EPS; maxPT_5.z = minPT_5.z + side_z / 2 + EPS;

		minPT_6.x = minPT_6.x + side_x / 2 - EPS; minPT_6.y = minPT_6.y + side_y / 2 - EPS;
		maxPT_6.x = minPT_6.x + side_x / 2 + EPS; maxPT_6.y = minPT_6.y + side_y / 2 + EPS; maxPT_6.z = minPT_6.z + side_z / 2 + EPS;

		minPT_7.y = minPT_7.y + side_y / 2 - EPS; minPT_7.z = minPT_7.z + side_z / 2 - EPS;
		maxPT_7.x = minPT_7.x + side_x / 2 + EPS; maxPT_7.y = minPT_7.y + side_y / 2 + EPS; maxPT_7.z = minPT_7.z + side_z / 2 + EPS;

		minPT_8.x = minPT_8.x + side_x / 2 - EPS; minPT_8.y = minPT_8.y + side_y / 2 - EPS; minPT_8.z = minPT_8.z + side_z / 2 - EPS;
		

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVoxelFilter1(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVoxelFilter2(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVoxelFilter3(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVoxelFilter4(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVoxelFilter5(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVoxelFilter6(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVoxelFilter7(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVoxelFilter8(new pcl::PointCloud<pcl::PointXYZ>);

		filter_based_on_bb(cloudIn, cloudVoxelFilter1, minPT_1, maxPT_1);
		filter_based_on_bb(cloudIn, cloudVoxelFilter2, minPT_2, maxPT_2);
		filter_based_on_bb(cloudIn, cloudVoxelFilter3, minPT_3, maxPT_3);
		filter_based_on_bb(cloudIn, cloudVoxelFilter4, minPT_4, maxPT_4);
		filter_based_on_bb(cloudIn, cloudVoxelFilter5, minPT_5, maxPT_5);
		filter_based_on_bb(cloudIn, cloudVoxelFilter6, minPT_6, maxPT_6);
		filter_based_on_bb(cloudIn, cloudVoxelFilter7, minPT_7, maxPT_7);
		filter_based_on_bb(cloudIn, cloudVoxelFilter8, minPT_8, maxPT_8);

		cloudVoxelFilter1 = voxel_grid_filter(cloudVoxelFilter1, leafSize);
		cloudVoxelFilter2 = voxel_grid_filter(cloudVoxelFilter2, leafSize);
		cloudVoxelFilter3 = voxel_grid_filter(cloudVoxelFilter3, leafSize);
		cloudVoxelFilter4 = voxel_grid_filter(cloudVoxelFilter4, leafSize);
		cloudVoxelFilter5 = voxel_grid_filter(cloudVoxelFilter5, leafSize);
		cloudVoxelFilter6 = voxel_grid_filter(cloudVoxelFilter6, leafSize);
		cloudVoxelFilter7 = voxel_grid_filter(cloudVoxelFilter7, leafSize);
		cloudVoxelFilter8 = voxel_grid_filter(cloudVoxelFilter8, leafSize);


		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZ>);
		*cloudOut = *cloudVoxelFilter1;
		*cloudOut += *cloudVoxelFilter2;
		*cloudOut += *cloudVoxelFilter3;
		*cloudOut += *cloudVoxelFilter4;
		*cloudOut += *cloudVoxelFilter5;
		*cloudOut += *cloudVoxelFilter6;
		*cloudOut += *cloudVoxelFilter7;
		*cloudOut += *cloudVoxelFilter8;
		return cloudOut;
		
	}
	else 
	{
		// Perform voxel grid filtering
		pcl::VoxelGrid<pcl::PointXYZ> vGF1;
		vGF1.setInputCloud(cloudIn);
		vGF1.setLeafSize(leafSize, leafSize, leafSize);
		vGF1.filter(*cloudIn);
		return cloudIn;
	}
	
}



void split_point_clouds_into_tiles(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOne,
									pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTwo,
									pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOneOverlap,
									pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTwoOverlap,
									pcl::PointXYZ bbMin,
								    pcl::PointXYZ bbMax,
								    int maxPointsPerTile,
									int minPointsPerTile,
									int &tileCounter,
								    int projectionDirection,
									float overlap,
									std::string saveFilePrefix)
{
	// Estimate how many splits will have to be performed minimaly (add one as division yields floor)
	int numberPoints = std::max(cloudOne->size(), cloudTwo->size());
	int minNumberOfTiles = numberPoints / maxPointsPerTile + 1;

	if (minNumberOfTiles == 1) 
	{
		// Ignore tiles with fewer than 1000 pts
		if (std::min(cloudOne->size(), cloudTwo->size()) > 1000){

		pcl::PLYWriter writer;
		writer.write(saveFilePrefix + "/tiled_data/source_tile_" + std::to_string(tileCounter) + ".ply", *cloudOne, true, false);
		writer.write(saveFilePrefix + "/tiled_data/target_tile_" + std::to_string(tileCounter) + ".ply", *cloudTwo, true, false);

		writer.write(saveFilePrefix + "/tiled_data/overlap/source_tile_" + std::to_string(tileCounter) + "_overlap.ply", *cloudOneOverlap, true, false);
		writer.write(saveFilePrefix + "/tiled_data/overlap/target_tile_" + std::to_string(tileCounter) + "_overlap.ply", *cloudTwoOverlap, true, false);

		tileCounter++;
		}
	}
	else 
	{
		// check in which direction the point cloud is projected
		if (projectionDirection == 0) 
		{
		
			float side_1 = bbMax.y - bbMin.y;
			float side_2 = bbMax.z - bbMin.z;

			if (side_1 > side_2) 
			{
				pcl::PointXYZ minPT_1 = bbMin;
				pcl::PointXYZ minPT_2 = bbMin;
				pcl::PointXYZ maxPT_1 = bbMax;
				pcl::PointXYZ maxPT_2 = bbMax;
				
				pcl::PointXYZ minPT_1_overlap = bbMin;
				pcl::PointXYZ minPT_2_overlap = bbMin;
				pcl::PointXYZ maxPT_1_overlap = bbMax;
				pcl::PointXYZ maxPT_2_overlap = bbMax;


				// Filter out part 1
				minPT_1.y = maxPT_1.y - side_1 / 2 - EPS;

				minPT_1_overlap.y = maxPT_1_overlap.y - side_1 / 2 - EPS - 20;
				maxPT_1_overlap.y = maxPT_1_overlap.y + 20;
				
				minPT_1_overlap.z = minPT_1_overlap.z - 20;
				maxPT_1_overlap.z = maxPT_1_overlap.z + 20;



				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_1(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_1(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_1_overlap(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_1_overlap(new pcl::PointCloud<pcl::PointXYZ>);

				filter_based_on_bb(cloudOne, cloudEpoch1_1, minPT_1, maxPT_1);
				filter_based_on_bb(cloudTwo, cloudEpoch2_1, minPT_1, maxPT_1);

				filter_based_on_bb(cloudOneOverlap, cloudEpoch1_1_overlap, minPT_1_overlap, maxPT_1_overlap);
				filter_based_on_bb(cloudTwoOverlap, cloudEpoch2_1_overlap, minPT_1_overlap, maxPT_1_overlap);

				// Filter out part 2
				maxPT_2.y = maxPT_2.y - side_1 / 2 + EPS;

				minPT_2_overlap.y = minPT_2_overlap.y - 20;
				maxPT_2_overlap.y = maxPT_2_overlap.y - side_1 / 2 + EPS + 20;
				
				minPT_2_overlap.z = minPT_2_overlap.z - 20;
				maxPT_2_overlap.z = maxPT_2_overlap.z + 20;


				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_2(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_2(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_2_overlap(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_2_overlap(new pcl::PointCloud<pcl::PointXYZ>);

				filter_based_on_bb(cloudOne, cloudEpoch1_2, minPT_2, maxPT_2);
				filter_based_on_bb(cloudTwo, cloudEpoch2_2, minPT_2, maxPT_2);

				filter_based_on_bb(cloudOneOverlap, cloudEpoch1_2_overlap, minPT_2_overlap, maxPT_2_overlap);
				filter_based_on_bb(cloudTwoOverlap, cloudEpoch2_2_overlap, minPT_2_overlap, maxPT_2_overlap);

				// Split the point clouds
				split_point_clouds_into_tiles(cloudEpoch1_1, cloudEpoch2_1,cloudEpoch1_1_overlap,cloudEpoch2_1_overlap, minPT_1, maxPT_1, maxPointsPerTile, minPointsPerTile, tileCounter, projectionDirection, overlap, saveFilePrefix);
				split_point_clouds_into_tiles(cloudEpoch1_2, cloudEpoch2_2, cloudEpoch1_2_overlap,cloudEpoch2_2_overlap, minPT_2, maxPT_2, maxPointsPerTile, minPointsPerTile, tileCounter, projectionDirection, overlap, saveFilePrefix);
			
			}
			else 
			{
				pcl::PointXYZ minPT_1 = bbMin;
				pcl::PointXYZ minPT_2 = bbMin;
				pcl::PointXYZ maxPT_1 = bbMax;
				pcl::PointXYZ maxPT_2 = bbMax;
				
				pcl::PointXYZ minPT_1_overlap = bbMin;
				pcl::PointXYZ minPT_2_overlap = bbMin;
				pcl::PointXYZ maxPT_1_overlap = bbMax;
				pcl::PointXYZ maxPT_2_overlap = bbMax;

				// Filter out part 1
				minPT_1.z = maxPT_1.z - side_2 / 2 - EPS;

				minPT_1_overlap.y = minPT_1_overlap.y - 20;
				maxPT_1_overlap.y = maxPT_1_overlap.y + 20;

				minPT_1_overlap.z = maxPT_1_overlap.z - side_2 / 2 - EPS - 20;
				maxPT_1_overlap.z = maxPT_1_overlap.z + 20;
				


				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_1(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_1(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_1_overlap(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_1_overlap(new pcl::PointCloud<pcl::PointXYZ>);

				filter_based_on_bb(cloudOne, cloudEpoch1_1, minPT_1, maxPT_1);
				filter_based_on_bb(cloudTwo, cloudEpoch2_1, minPT_1, maxPT_1);

				filter_based_on_bb(cloudOneOverlap, cloudEpoch1_1_overlap, minPT_1_overlap, maxPT_1_overlap);
				filter_based_on_bb(cloudTwoOverlap, cloudEpoch2_1_overlap, minPT_1_overlap, maxPT_1_overlap);


				// Filter out part 2
				maxPT_2.z = maxPT_2.z - side_2 / 2 + EPS;

				minPT_2_overlap.y = minPT_2_overlap.y - 20;
				maxPT_2_overlap.y = maxPT_2_overlap.y + 20;

				minPT_2_overlap.z = minPT_2_overlap.z - 20;
				maxPT_2_overlap.z = maxPT_2_overlap.z - side_2 / 2 + EPS + 20;

				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_2(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_2(new pcl::PointCloud<pcl::PointXYZ>);
				
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_2_overlap(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_2_overlap(new pcl::PointCloud<pcl::PointXYZ>);
				
				filter_based_on_bb(cloudOne, cloudEpoch1_2, minPT_2, maxPT_2);
				filter_based_on_bb(cloudTwo, cloudEpoch2_2, minPT_2, maxPT_2);

				filter_based_on_bb(cloudOneOverlap, cloudEpoch1_2_overlap, minPT_2_overlap, maxPT_2_overlap);
				filter_based_on_bb(cloudTwoOverlap, cloudEpoch2_2_overlap, minPT_2_overlap, maxPT_2_overlap);

				// Split the point clouds
				split_point_clouds_into_tiles(cloudEpoch1_1, cloudEpoch2_1,cloudEpoch1_1_overlap,cloudEpoch2_1_overlap, minPT_1, maxPT_1, maxPointsPerTile, minPointsPerTile, tileCounter, projectionDirection, overlap, saveFilePrefix);
				split_point_clouds_into_tiles(cloudEpoch1_2, cloudEpoch2_2, cloudEpoch1_2_overlap,cloudEpoch2_2_overlap, minPT_2, maxPT_2, maxPointsPerTile, minPointsPerTile, tileCounter, projectionDirection, overlap, saveFilePrefix);			
			}

		}

		else if (projectionDirection == 1) 
		{
			float side_1 = bbMax.x - bbMin.x;
			float side_2 = bbMax.z - bbMin.z;

			if (side_1 > side_2)
			{
				pcl::PointXYZ minPT_1 = bbMin;
				pcl::PointXYZ minPT_2 = bbMin;
				pcl::PointXYZ maxPT_1 = bbMax;
				pcl::PointXYZ maxPT_2 = bbMax;

				pcl::PointXYZ minPT_1_overlap = bbMin;
				pcl::PointXYZ minPT_2_overlap = bbMin;
				pcl::PointXYZ maxPT_1_overlap = bbMax;
				pcl::PointXYZ maxPT_2_overlap = bbMax;

				// Filter out part 1
				minPT_1.x = maxPT_1.x - side_1 / 2 - EPS - overlap;

				minPT_1_overlap.x = maxPT_1_overlap.x - side_1 / 2 - EPS - 20;
				maxPT_1_overlap.x = maxPT_1_overlap.x + 20;
				
				minPT_1_overlap.z = minPT_1_overlap.z - 20;
				maxPT_1_overlap.z = maxPT_1_overlap.z + 20;

				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_1(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_1(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_1_overlap(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_1_overlap(new pcl::PointCloud<pcl::PointXYZ>);

				filter_based_on_bb(cloudOne, cloudEpoch1_1, minPT_1, maxPT_1);
				filter_based_on_bb(cloudTwo, cloudEpoch2_1, minPT_1, maxPT_1);

				filter_based_on_bb(cloudOneOverlap, cloudEpoch1_1_overlap, minPT_1_overlap, maxPT_1_overlap);
				filter_based_on_bb(cloudTwoOverlap, cloudEpoch2_1_overlap, minPT_1_overlap, maxPT_1_overlap);

				// Filter out part 2
				maxPT_2.x = maxPT_2.x - side_1 / 2 + EPS;

				minPT_2_overlap.x = minPT_2_overlap.x - 20;
				maxPT_2_overlap.x = maxPT_2_overlap.x - side_1 / 2 + EPS + 20;
				
				minPT_2_overlap.z = minPT_2_overlap.z - 20;
				maxPT_2_overlap.z = maxPT_2_overlap.z + 20;


				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_2(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_2(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_2_overlap(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_2_overlap(new pcl::PointCloud<pcl::PointXYZ>);

				filter_based_on_bb(cloudOne, cloudEpoch1_2, minPT_2, maxPT_2);
				filter_based_on_bb(cloudTwo, cloudEpoch2_2, minPT_2, maxPT_2);

				filter_based_on_bb(cloudOneOverlap, cloudEpoch1_2_overlap, minPT_2_overlap, maxPT_2_overlap);
				filter_based_on_bb(cloudTwoOverlap, cloudEpoch2_2_overlap, minPT_2_overlap, maxPT_2_overlap);

				// Split the point clouds
				split_point_clouds_into_tiles(cloudEpoch1_1, cloudEpoch2_1,cloudEpoch1_1_overlap,cloudEpoch2_1_overlap, minPT_1, maxPT_1, maxPointsPerTile, minPointsPerTile, tileCounter, projectionDirection, overlap, saveFilePrefix);
				split_point_clouds_into_tiles(cloudEpoch1_2, cloudEpoch2_2, cloudEpoch1_2_overlap,cloudEpoch2_2_overlap, minPT_2, maxPT_2, maxPointsPerTile, minPointsPerTile, tileCounter, projectionDirection, overlap, saveFilePrefix);
			
			}
			else
			{
				pcl::PointXYZ minPT_1 = bbMin;
				pcl::PointXYZ minPT_2 = bbMin;
				pcl::PointXYZ maxPT_1 = bbMax;
				pcl::PointXYZ maxPT_2 = bbMax;


				pcl::PointXYZ minPT_1_overlap = bbMin;
				pcl::PointXYZ minPT_2_overlap = bbMin;
				pcl::PointXYZ maxPT_1_overlap = bbMax;
				pcl::PointXYZ maxPT_2_overlap = bbMax;

				// Filter out part 1
				minPT_1.z = maxPT_1.z - side_2 / 2 - EPS;

				minPT_1_overlap.x = minPT_1_overlap.x - 20;
				maxPT_1_overlap.x = maxPT_1_overlap.x + 20;

				minPT_1_overlap.z = maxPT_1_overlap.z - side_2 / 2 - EPS - 20;
				maxPT_1_overlap.z = maxPT_1_overlap.z + 20;
				
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_1(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_1(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_1_overlap(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_1_overlap(new pcl::PointCloud<pcl::PointXYZ>);

				filter_based_on_bb(cloudOne, cloudEpoch1_1, minPT_1, maxPT_1);
				filter_based_on_bb(cloudTwo, cloudEpoch2_1, minPT_1, maxPT_1);

				filter_based_on_bb(cloudOneOverlap, cloudEpoch1_1_overlap, minPT_1_overlap, maxPT_1_overlap);
				filter_based_on_bb(cloudTwoOverlap, cloudEpoch2_1_overlap, minPT_1_overlap, maxPT_1_overlap);


				// Filter out part 2
				maxPT_2.z = maxPT_2.z - side_2 / 2 + EPS;

				minPT_2_overlap.x = minPT_2_overlap.x - 20;
				maxPT_2_overlap.x = maxPT_2_overlap.x + 20;

				minPT_2_overlap.z = minPT_2_overlap.z - 20;
				maxPT_2_overlap.z = maxPT_2_overlap.z - side_2 / 2 + EPS + 20;

				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_2(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_2(new pcl::PointCloud<pcl::PointXYZ>);
				
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_2_overlap(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_2_overlap(new pcl::PointCloud<pcl::PointXYZ>);
				
				filter_based_on_bb(cloudOne, cloudEpoch1_2, minPT_2, maxPT_2);
				filter_based_on_bb(cloudTwo, cloudEpoch2_2, minPT_2, maxPT_2);

				filter_based_on_bb(cloudOneOverlap, cloudEpoch1_2_overlap, minPT_2_overlap, maxPT_2_overlap);
				filter_based_on_bb(cloudTwoOverlap, cloudEpoch2_2_overlap, minPT_2_overlap, maxPT_2_overlap);

				// Split the point clouds
				split_point_clouds_into_tiles(cloudEpoch1_1, cloudEpoch2_1,cloudEpoch1_1_overlap,cloudEpoch2_1_overlap, minPT_1, maxPT_1, maxPointsPerTile, minPointsPerTile, tileCounter, projectionDirection, overlap, saveFilePrefix);
				split_point_clouds_into_tiles(cloudEpoch1_2, cloudEpoch2_2, cloudEpoch1_2_overlap,cloudEpoch2_2_overlap, minPT_2, maxPT_2, maxPointsPerTile, minPointsPerTile, tileCounter, projectionDirection, overlap, saveFilePrefix);			
			}

		}
		
		else
		{
			float side_1 = bbMax.x - bbMin.x;
			float side_2 = bbMax.y - bbMin.y;

			if (side_1 > side_2)
			{
				pcl::PointXYZ minPT_1 = bbMin;
				pcl::PointXYZ minPT_2 = bbMin;
				pcl::PointXYZ maxPT_1 = bbMax;
				pcl::PointXYZ maxPT_2 = bbMax;

				pcl::PointXYZ minPT_1_overlap = bbMin;
				pcl::PointXYZ minPT_2_overlap = bbMin;
				pcl::PointXYZ maxPT_1_overlap = bbMax;
				pcl::PointXYZ maxPT_2_overlap = bbMax;

				// Filter out part 1
				minPT_1.x = maxPT_1.x - side_1 / 2 - EPS;
				
				minPT_1_overlap.x = maxPT_1_overlap.x - side_1 / 2 - EPS - 20;
				maxPT_1_overlap.x = maxPT_1_overlap.x + 20;

				minPT_1_overlap.y = minPT_1_overlap.y - 20;
				maxPT_1_overlap.y = maxPT_1_overlap.y + 20;


				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_1(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_1(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_1_overlap(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_1_overlap(new pcl::PointCloud<pcl::PointXYZ>);

				filter_based_on_bb(cloudOne, cloudEpoch1_1, minPT_1, maxPT_1);
				filter_based_on_bb(cloudTwo, cloudEpoch2_1, minPT_1, maxPT_1);

				filter_based_on_bb(cloudOneOverlap, cloudEpoch1_1_overlap, minPT_1_overlap, maxPT_1_overlap);
				filter_based_on_bb(cloudTwoOverlap, cloudEpoch2_1_overlap, minPT_1_overlap, maxPT_1_overlap);

				// Filter out part 2
				maxPT_2.x = maxPT_2.x - side_1 / 2 + EPS;

				minPT_2_overlap.x = minPT_2_overlap.x - 20;
				maxPT_2_overlap.x = maxPT_2_overlap.x - side_1 / 2 - EPS + 20;

				minPT_2_overlap.y = minPT_2_overlap.y - 20;
				maxPT_2_overlap.y = maxPT_2_overlap.y + 20;
				
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_2(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_2(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_2_overlap(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_2_overlap(new pcl::PointCloud<pcl::PointXYZ>);

				filter_based_on_bb(cloudOne, cloudEpoch1_2, minPT_2, maxPT_2);
				filter_based_on_bb(cloudTwo, cloudEpoch2_2, minPT_2, maxPT_2);

				filter_based_on_bb(cloudOneOverlap, cloudEpoch1_2_overlap, minPT_2_overlap, maxPT_2_overlap);
				filter_based_on_bb(cloudTwoOverlap, cloudEpoch2_2_overlap, minPT_2_overlap, maxPT_2_overlap);

				// Split the point clouds
				split_point_clouds_into_tiles(cloudEpoch1_1, cloudEpoch2_1,cloudEpoch1_1_overlap,cloudEpoch2_1_overlap, minPT_1, maxPT_1, maxPointsPerTile, minPointsPerTile, tileCounter, projectionDirection, overlap, saveFilePrefix);
				split_point_clouds_into_tiles(cloudEpoch1_2, cloudEpoch2_2, cloudEpoch1_2_overlap,cloudEpoch2_2_overlap, minPT_2, maxPT_2, maxPointsPerTile, minPointsPerTile, tileCounter, projectionDirection, overlap, saveFilePrefix);
			
			}
			else
			{
				pcl::PointXYZ minPT_1 = bbMin;
				pcl::PointXYZ minPT_2 = bbMin;
				pcl::PointXYZ maxPT_1 = bbMax;
				pcl::PointXYZ maxPT_2 = bbMax;

				pcl::PointXYZ minPT_1_overlap = bbMin;
				pcl::PointXYZ minPT_2_overlap = bbMin;
				pcl::PointXYZ maxPT_1_overlap = bbMax;
				pcl::PointXYZ maxPT_2_overlap = bbMax;

				// Filter out part 1
				minPT_1.y = maxPT_1.y - side_2 / 2 - EPS;

				minPT_1_overlap.x = minPT_1_overlap.x - 20;
				maxPT_1_overlap.x = maxPT_1_overlap.x + 20;

				minPT_1_overlap.y = maxPT_1_overlap.y - side_2 / 2 - EPS - 20;
				maxPT_1_overlap.y = maxPT_1_overlap.y + 20;
				
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_1(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_1(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_1_overlap(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_1_overlap(new pcl::PointCloud<pcl::PointXYZ>);

				filter_based_on_bb(cloudOne, cloudEpoch1_1, minPT_1, maxPT_1);
				filter_based_on_bb(cloudTwo, cloudEpoch2_1, minPT_1, maxPT_1);

				filter_based_on_bb(cloudOneOverlap, cloudEpoch1_1_overlap, minPT_1_overlap, maxPT_1_overlap);
				filter_based_on_bb(cloudTwoOverlap, cloudEpoch2_1_overlap, minPT_1_overlap, maxPT_1_overlap);



				// Filter out part 2
				maxPT_2.y = maxPT_2.y - side_2 / 2 + EPS;

				minPT_2_overlap.x = minPT_2_overlap.x - 20;
				maxPT_2_overlap.x = maxPT_2_overlap.x + 20;

				minPT_2_overlap.y = minPT_2_overlap.y - 20;
				maxPT_2_overlap.y = maxPT_2_overlap.y - side_2 / 2 + EPS + 20;

				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_2(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_2(new pcl::PointCloud<pcl::PointXYZ>);
				
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1_2_overlap(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2_2_overlap(new pcl::PointCloud<pcl::PointXYZ>);
				
				filter_based_on_bb(cloudOne, cloudEpoch1_2, minPT_2, maxPT_2);
				filter_based_on_bb(cloudTwo, cloudEpoch2_2, minPT_2, maxPT_2);

				filter_based_on_bb(cloudOneOverlap, cloudEpoch1_2_overlap, minPT_2_overlap, maxPT_2_overlap);
				filter_based_on_bb(cloudTwoOverlap, cloudEpoch2_2_overlap, minPT_2_overlap, maxPT_2_overlap);

				// Split the point clouds
				split_point_clouds_into_tiles(cloudEpoch1_1, cloudEpoch2_1,cloudEpoch1_1_overlap,cloudEpoch2_1_overlap, minPT_1, maxPT_1, maxPointsPerTile, minPointsPerTile, tileCounter, projectionDirection, overlap, saveFilePrefix);
				split_point_clouds_into_tiles(cloudEpoch1_2, cloudEpoch2_2, cloudEpoch1_2_overlap,cloudEpoch2_2_overlap, minPT_2, maxPT_2, maxPointsPerTile, minPointsPerTile, tileCounter, projectionDirection, overlap, saveFilePrefix);			
			}
		}	
	}

}


bool is_file_exist(const std::string fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}

bool resave_point_cloud(std::string firstPointCloud, 
					   std::string secondPointCloud,
					   bool verbose)
{
	// Initialize the point clouds
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2(new pcl::PointCloud<pcl::PointXYZ>);

	if (verbose){std::cout << "Reading point cloud data." << std::endl;}
	

	if (is_file_exist(firstPointCloud))
	{
		pcl::io::loadPLYFile(firstPointCloud, *cloudEpoch1);
		if (verbose){
		std::cout << "Point cloud 1 read in!" << std::endl;
		std::cout << "Number of Points: " << cloudEpoch1->size() << std::endl;
		}
	}
	else
	{
		std::cout << "File " << firstPointCloud << " does not exist!!!" << std::endl;
		return false;
	}

	// Read point cloud epoch 2 
	if (is_file_exist(secondPointCloud))
	{
		if (verbose){
		pcl::io::loadPLYFile(secondPointCloud, *cloudEpoch2);
		std::cout << "Point cloud 2 read in!" << std::endl;
		std::cout << "Number of Points: " << cloudEpoch2->size() << std::endl;
		}
	}
	else
	{
		std::cout <<"ERROR:" << "File " << secondPointCloud << " does not exist!!!" << std::endl;
		return false;
	}
	
	// Save point clouds
	pcl::PLYWriter writer;
	writer.write(firstPointCloud, *cloudEpoch1, true, false);
	writer.write(secondPointCloud, *cloudEpoch2, true, false);
}


bool tile_point_clouds(std::string firstPointCloud, 
					   std::string secondPointCloud,
					   int maxPointsPerTile, 
					   int minPointsPerTile,
					   bool voxelGridFlag,
					   float voxelGridFilterSize, 
					   float overlapTiles,
					   int projectionDirection,
					   bool verbose)
{
	// Initialize the point clouds
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudEpoch2(new pcl::PointCloud<pcl::PointXYZ>);

	if (verbose){std::cout << "Reading point cloud data." << std::endl;}

	if (is_file_exist(firstPointCloud))
	{
		pcl::io::loadPLYFile(firstPointCloud, *cloudEpoch1);
		if (verbose){
		std::cout << "Point cloud 1 read in!" << std::endl;
		std::cout << "Number of Points: " << cloudEpoch1->size() << std::endl;
		}
	}
	else
	{
		std::cout << "File " << firstPointCloud << " does not exist!!!" << std::endl;
		return false;
	}

	// Read point cloud epoch 2 
	if (is_file_exist(secondPointCloud))
	{
		pcl::io::loadPLYFile(secondPointCloud, *cloudEpoch2);
		if (verbose){
		std::cout << "Point cloud 2 read in!" << std::endl;
		std::cout << "Number of Points: " << cloudEpoch2->size() << std::endl;
		}
	}
	else
	{
		std::cout <<"ERROR:" << "File " << secondPointCloud << " does not exist!!!" << std::endl;
		return false;
	}
	
	// Extract the folder path for saving the data
	std::size_t found = firstPointCloud.find_last_of("/\\");
	std::string tempFileName = firstPointCloud.substr(0, found);
	found = tempFileName.find_last_of("/\\");
	std::string saveFilePrefix = tempFileName.substr(0, found);


	// Get the bounding box of the first point cloud
	pcl::PointXYZ minPtCloud1, maxPtCloud1;
	pcl::getMinMax3D(*cloudEpoch1, minPtCloud1, maxPtCloud1);

	// Get the bounding box of the second point cloud
	pcl::PointXYZ minPtCloud2, maxPtCloud2;
	pcl::getMinMax3D(*cloudEpoch2, minPtCloud2, maxPtCloud2);
	
	pcl::PointXYZ bbOverlapMin, bbOverlapMax;
	std::vector<float> bbOverlapArea;
	overlap_bounding_boxes(minPtCloud1, maxPtCloud1, minPtCloud2, maxPtCloud2, bbOverlapMin, bbOverlapMax, bbOverlapArea);
	

	// Filter the point clouds based on the overlaping bounding box 

	filter_based_on_bb(cloudEpoch1, cloudEpoch1, bbOverlapMin, bbOverlapMax);
	filter_based_on_bb(cloudEpoch2, cloudEpoch2, bbOverlapMin, bbOverlapMax);

	if (verbose)
	{
	std::cout << "-----------------------------------------------------------------------------" << std::endl;
	std::cout << "Starting voxel grid filter!" << std::endl;
	}

	// Create folder for saving data
	boost::filesystem::path dstFolder = saveFilePrefix + "/tiled_data/";
	boost::filesystem::create_directory(dstFolder);

	boost::filesystem::path dstFolder_overlap = saveFilePrefix + "/tiled_data/overlap/";
	boost::filesystem::create_directory(dstFolder_overlap);

	// Filter the point clouds using a voxel grid filter in order to make the resolution more uniform
	
	if (voxelGridFlag){
		if (voxelGridFilterSize == 0.0) 
		{
			if (cloudEpoch1->size() < cloudEpoch2->size())
				voxelGridFilterSize = median_point_cloud_resolution(cloudEpoch1);
			else
				voxelGridFilterSize = median_point_cloud_resolution(cloudEpoch2);

			if (verbose){std::cout << "Size of the filter: " << voxelGridFilterSize << "m determined based on the median resolution!" << std::endl;}
		}
		else
			if (verbose){std::cout << "Using predifend size of the filter: " << voxelGridFilterSize << "m." << std::endl;}

		cloudEpoch1 = voxel_grid_filter(cloudEpoch1, voxelGridFilterSize);
		// pcl::io::savePLYFileBinary("01_Data/voxelGridEpoch1.ply", *cloudEpoch1);
		if (verbose)
		{
		std::cout << "Point cloud one complete! " << cloudEpoch1->size() << " points remaining after voxel grid filter." << std::endl;
		}
		cloudEpoch2 = voxel_grid_filter(cloudEpoch2, voxelGridFilterSize);
		// pcl::io::savePLYFileBinary("01_Data/voxelGridEpoch2.ply", *cloudEpoch2);
		if (verbose)
		{
		std::cout << "Point cloud two complete! " << cloudEpoch2->size() << " points remaining after voxel grid filter." << std::endl;
		std::cout << "-----------------------------------------------------------------------------" << std::endl;
		}
	
	}
	
	// Find the area with the biggest surface to project the point to
	if (projectionDirection == -1)
		projectionDirection = std::distance(bbOverlapArea.begin(), std::max_element(bbOverlapArea.begin(), bbOverlapArea.end()));


	// Estimate how many splits will have to be performed minimaly (add one as division yields floor)
	int minNumberOfTiles = std::max(cloudEpoch1->size(), cloudEpoch2->size()) / maxPointsPerTile + 1;

	if (verbose)
	{
	std::cout << "Spliting point clouds into patches!" << std::endl;
	std::cout << "Maximum number of points in a single patch: " << maxPointsPerTile << std::endl;
	std::cout << "More then " << minNumberOfTiles << " patches per epoch will be generated." << std::endl;
	std::vector<std::string> projection = { "X","Y","Z" };
	std::cout << "Point clouds will be projected along the " << projection[projectionDirection] << " axis." << std::endl;
	}
	
	int tileCounter = 0;

	split_point_clouds_into_tiles(cloudEpoch1, cloudEpoch2, cloudEpoch1, cloudEpoch2, bbOverlapMin, bbOverlapMax, maxPointsPerTile, minPointsPerTile, tileCounter, projectionDirection, overlapTiles, saveFilePrefix);

	if (verbose)
	{
	std::cout << "Spliting complete. " << tileCounter +1 << " patches saved per epoch." << std::endl;
	std::cout << "-----------------------------------------------------------------------------" << std::endl;
	}
	return true;
}

#include "codelibrary/base/log.h"
#include "codelibrary/geometry/io/xyz_io.h"
#include "codelibrary/geometry/point_cloud/pca_estimate_normals.h"
#include "codelibrary/geometry/point_cloud/supervoxel_segmentation.h"
#include "codelibrary/geometry/util/distance_3d.h"
#include "codelibrary/util/tree/kd_tree.h"
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
//#include <pcl/impl/point_types.hpp>
#include "supervoxel.h"

/// Point with Normal.
struct PointWithNormal : cl::RPoint3D {
    PointWithNormal() {}

    cl::RVector3D normal;
};

/**
 * Metric used in VCCS supervoxel segmentation.
 *
 * Reference:
 *   Rusu, R.B., Cousins, S., 2011. 3d is here: Point cloud library (pcl),
 *   IEEE International Conference on Robotics and Automation, pp. 1–4.
 */
class VCCSMetric {
public:
    explicit VCCSMetric(double resolution)
        : resolution_(resolution) {}

    double operator() (const PointWithNormal& p1,
                       const PointWithNormal& p2) const {
        return 1.0 - std::fabs(p1.normal * p2.normal) +
               cl::geometry::Distance(p1, p2) / resolution_ * 0.4;
    }

private:
    double resolution_;
};

/**
 * Save point clouds (with segmentation colors) into the file.
 */
void WritePoints(const char* filename,
                 int n_supervoxels,
                 const cl::Array<cl::RPoint3D>& points,
                 const cl::Array<int>& labels) {
    cl::Array<cl::RGB32Color> colors(points.size());
    std::mt19937 random;
    cl::Array<cl::RGB32Color> supervoxel_colors(n_supervoxels);
    for (int i = 0; i < n_supervoxels; ++i) {
        supervoxel_colors[i] = cl::RGB32Color(random());
    }
    for (int i = 0; i < points.size(); ++i) {
        colors[i] = supervoxel_colors[labels[i]];
    }

    if (cl::geometry::io::WriteXYZPoints(filename, points, colors, labels)) {
        LOG(INFO) << "The points are written into " << filename;
    }

//    system(filename);
}

void pclToXYZArray(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, cl::Array<cl::Point3D<double> >* points)
{
    // Iterrate over all the points 
    int number_of_points = cloud->points.size();
    double x, y, z;

    for (int i = 0; i < number_of_points; i++)
    {
        x = cloud->points[i].x;
        y = cloud->points[i].y;
        z = cloud->points[i].z;

        // Insert point to the array
        points->emplace_back(x, y, z);
    }
}

std::vector<int> computeSupervoxel(std::string input_file, int k_neighbors, double resolution, std::string save_file = "None")
{
    cl::Array<cl::RPoint3D> points;
    cl::Array<cl::RGB32Color> colors;

    // Read in the point cloud from the filename
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPLYFile(input_file, *cloud);

    pclToXYZArray(cloud, &points);

    int n_points = points.size();

    cl::KDTree<cl::RPoint3D> kdtree;
    kdtree.SwapPoints(&points);


    assert(k_neighbors < n_points);

    cl::Array<cl::RVector3D> normals(n_points);
    cl::Array<cl::Array<int> > neighbors(n_points);
    cl::Array<cl::RPoint3D> neighbor_points(k_neighbors);
    for (int i = 0; i < n_points; ++i) {
        kdtree.FindKNearestNeighbors(kdtree.points()[i], k_neighbors,
                                     &neighbors[i]);
        for (int k = 0; k < k_neighbors; ++k) {
            neighbor_points[k] = kdtree.points()[neighbors[i][k]];
        }
        cl::geometry::point_cloud::PCAEstimateNormal(neighbor_points.begin(),
                                                     neighbor_points.end(),
                                                     &normals[i]);
    }
    kdtree.SwapPoints(&points);

    cl::Array<PointWithNormal> oriented_points(n_points);
    for (int i = 0; i < n_points; ++i) {
        oriented_points[i].x = points[i].x;
        oriented_points[i].y = points[i].y;
        oriented_points[i].z = points[i].z;
        oriented_points[i].normal = normals[i];
    }


    VCCSMetric metric(resolution);
    cl::Array<int> labels, supervoxels;
    cl::geometry::point_cloud::SupervoxelSegmentation(oriented_points,
                                                      neighbors,
                                                      resolution,
                                                      metric,
                                                      &supervoxels,
                                                      &labels);

    int n_supervoxels = supervoxels.size();

    if (save_file != "None"){
        const char* save_file_char = save_file.c_str();
        WritePoints(save_file_char, n_supervoxels, points, labels);}

    std::vector<int> output_labels(labels.begin(), labels.end());
    return output_labels;
} 

int main() {
    LOG_ON(INFO);

    const std::string filename = "data/raw_data/test.ply";
    int k_neighbors = 15;
    double resolution = 1.0;
    std::string save_file = "test.xyz"; 

    computeSupervoxel(filename, k_neighbors, resolution, save_file);
    return 0;
}

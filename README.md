This repository contains the source code and instructions of F2S3, an approach for deep learning based deformation monitoring of landslides and rockfalls using point cloud data. It accompanies the recent publication:

## F2S3: Robustified determination of 3D displacement vector fields using deep learning

**|[PDF](https://www.degruyter.com/view/journals/jag/14/2/article-p177.xml)  | [Group Page](https://gseg.igp.ethz.ch/)|**

**| [Zan Gojcic](https://zgojcic.github.io/) |
[Caifa Zhou](https://caifazhou.github.io/) |
[Lorenz Schmid](https://gseg.igp.ethz.ch/people/scientific-assistance/lorenz-schmid.html) |
[Andreas Wieser](https://gseg.igp.ethz.ch/people/group-head/prof-dr--andreas-wieser.html) |**

Feature to Feature Supervoxel-based Spatial Smoothing (F2S3) is a deep learning based deformation analysis method for point cloud data. It computes a displacements vector field between two epochs, based on the establishing the corresponding points in the feature space. The initial noisy set of putative correspondences is filtered using an outlier detection network, which operates inside individual supervoxels and hence satisfies the local consistency constraint without crossing the discontinuities of the vector field. F2S3 achieves a very high performance in the point clouds with sufficient local structure and thus represents a complementary method to the traditional deformation analysis tools such as C2C, M2C, and M3C2.

![F2S3 pipeline](docs/assets/F2S3_pipeline.PNG?raw=true)

## Installation

F2S3 has been successfully tested on Ubuntu 20.04 LTS, 22.04 LTS, and 24.04 LTS, both as native OS and via Windows Subsystem for Linux (WSL2). The installation process involves setting up a Python environment, installing PyTorch with CUDA support, and installing necessary system dependencies for the C++ extensions.

The provided code is partially implemented in C++ (tiling and supervoxel segmentation) and partially in Python using Pytorch library (feature extraction and filtering). The code relies on CUDA implementation and therefore requires an NVIDIA graphic card with [CUDA support](https://developer.nvidia.com/cuda-gpus).

**Detailled installation instructions are available [here](docs/installation.md).**
**Development and container workflow instructions are available [here](docs/development.md).**

## Running F2S3

**Basic function call**

```shell
f2s3 --source_cloud ./data/_sample_folder/raw_data/epoch1.ply --target_cloud ./data/_sample_folder/raw_data/epoch2.ply
```

**More settings**
```shell
f2s3 --help
```

## Citation

If you find this code useful for your work or use it in your projects, please consider citing:

```shell
@article {gojcic2020F2S3,
      title = {F2S3: Robustified determination of 3D displacement vector fields using deep learning},
      author = {Gojcic, Zan and Zhou, Caifa and Wieser Andreas},
      journal = {Journal of Applied Geodesy},
      year = {2020},
      publisher = {De Gruyter},
      volume = {14},
      number = {2},
      pages= {177 - 189},
}
```

## Related Projects

- Outlier detection network: [3D Multiview registration, CVPR'20](https://github.com/zgojcic/3D_multiview_reg)
- Learned 3D local feature descriptor: [3DSmoothNet, CVPR'19](https://github.com/zgojcic/3DSmoothNet)
- Fast approximate nearest neighbor search: [HNSW, TPAMI'16](https://github.com/nmslib/hnswlib)
- Boundary preserving supervoxel segmentation: [Supervoxel Seg., ISPRS'18](https://github.com/yblin/Supervoxel-for-3D-point-clouds)

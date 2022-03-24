Point cloud tiling tool enables the tiling of two registered point clouds (two epochs) into smaller tiles that can be processed on a standalone computer. The current bottleneck of the approach are the input parametrization and the nearest neighbor search. Tiling of the point clouds is performed in 2D after projecting the point clouds along one of the axis. Each tile is subdivided into two subtiles along a larger dimensions (width or hight) until the conditions (max and min number of points) are satisfied. More information is available in **cite the landslides paper**. 

<pre><code>

<i>class</i> <b>pc_tiling.resave_point_cloud(firstPointCloud, secondPointCloud, verbose=False)</b>

<b>Parameters:</b> 

- <i>firstPointCloud (string):</i> path to the ".ply" file of the source epoch

- <i>secondPointCloud (string):</i> path to the ".ply" file of the target epoch

- <i>verbose (bool):</i> if selected detailed information will be written to the command line 



<i>class</i> <b>pc_tiling.tile_point_clouds(firstPointCloud, secondPointCloud, maxPointsPerTile = 1000000, minPointsPerTile = 100, voxelGridFlag = false, voxelGridFilterSize = 0.05, overlapTiles = 0.0, projectionDirection = -1, verbose=False)</b>

<b>Parameters:</b> 

- <i>firstPointCloud (string):</i> path to the ".ply" file of the source epoch

- <i>secondPointCloud (string):</i> path to the ".ply" file of the target epoch

- <i>maxPointsPerTile (int):</i> maximum number of points per tile (~10^6 point can be processed on 64GB of RAM)

- <i>minPointsPerTile (int):</i> minimum number of points per tile (if less points the point cloud tile will not be saved)

- <i>voxelGridFlag (bool):</i> flag for voxel grid filtering (uniforming the point cloud resolution)

- <i>voxelGridFilterSize (float):</i> if voxelGridFlag=True defines the size of the voxel grid filter (if =0.0, median point cloud resolution will be calculated and used as the filter size)

- <i>overlapTiles (float):</i> defines the intra-epochal overlap of the neighboring tiles

- <i>projectionDirection (float):</i> axis along which the point cloud will be projected (0=X, 1=Y, 2=Z, -1=axis with the largest projection area)

- <i>verbose (bool):</i> if selected detailed information will be written to the command line 

<b>Returns:</b>

- Function saves point of each tile into a separate "*.ply" file, such that the tile "source_tile_x.ply" corresponds to the "target_tile_x.ply", where "x" is the number of the tile.

</code></pre>
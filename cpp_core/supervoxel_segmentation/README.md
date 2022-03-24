Supervoxel segmentation tool enables the segmentation of a point cloud into boundary preserving super voxels. Boundary preservation is achieved by incorporating cos distance between the normal vectors in the objective function. This source code is based on a recent [ISPRS paper](https://www.sciencedirect.com/science/article/pii/S0924271618301370) and the provided [source code](https://github.com/yblin/Supervoxel-for-3D-point-clouds). Currently we only provide a python wrapper for a single function, but other functiones can be added by changing the `supervoxel.i` file.

<pre><code>
<i>class</i> <b>supervoxel.computeSupervoxel(input_file, k_neighbors, resolution, save_file)</b>

<b>Parameters:</b> 

- <i>input_file (string):</i> path to the file that should be segmented into supervoxels

- <i>k_neighbors (int):</i> number of nearest neighbors used for the normal vector estimation

- <i>resolution (float):</i> desired diameter of the supervoxels indirectly
  controls the number of supervoxels (see paper for more information)

- <i>save_file (string):</i> path where the results should be saved if "None" data will not be saved (the folder must exist already) 

<b>Returns:</b>

- Function saves "ASCII" files to the defined path (save_file). First three
  columns are the coordinates of the points, columns 3-6 are randomly assigned
  colors (based on label), label denoting the supervoxel (all points with the 
  same label belong to one supervoxel) 

</code></pre>
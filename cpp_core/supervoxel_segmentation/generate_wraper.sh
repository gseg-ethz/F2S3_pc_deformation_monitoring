cmake -DCMAKE_BUILD_TYPE=Release .
make -j 8
swig -c++ -python supervoxel.i
python -c "import supervoxel"

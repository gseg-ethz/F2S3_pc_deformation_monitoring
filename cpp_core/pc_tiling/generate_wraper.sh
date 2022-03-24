cmake -DCMAKE_BUILD_TYPE=Release .
make -j 8
swig -c++ -python pc_tiling.i
python -c "import pc_tiling"
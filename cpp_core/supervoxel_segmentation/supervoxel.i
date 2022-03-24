 /* Bonjour.i */
 %module supervoxel

 %{
	#define SWIG_FILE_WITH_INIT
 	#include "supervoxel.h"
 %}




 %include std_string.i
 %include std_vector.i

 %template(IntVector) std::vector<int>;

 %include "numpy.i"
 %init %{
 import_array();
 %}

 %include "supervoxel.h"



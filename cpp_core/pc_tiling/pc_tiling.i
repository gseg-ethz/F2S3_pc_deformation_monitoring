 /* Bonjour.i */
 %module pc_tiling

 %{
	#define SWIG_FILE_WITH_INIT
 	#include "pc_tiling.h"
 %}

 %include std_string.i
 %include std_vector.i

 // %include "numpy.i"

 %include "pc_tiling.h"



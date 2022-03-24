#include "Bonjour.hpp"


using namespace boost::python;

BOOST_PYTHON_MODULE(pylib)
{
    class_< Bonjour >("Bonjour", init<std::string>())
      .def("greet", &Bonjour::greet)
      .def("square", &Bonjour::square)
      .add_property("msg", &Bonjour::get_msg, &Bonjour::set_msg);
}


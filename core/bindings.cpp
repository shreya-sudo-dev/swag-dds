#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dsp_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(core, m)
{
    py::class_<DoAResult>(m, "DoAResult")
        .def_readonly("angle", &DoAResult::angle)
        .def_readonly("confident", &DoAResult::confident);

    py::class_<DSPEngine>(m, "DSPEngine")
        .def(py::init<>())

        .def("push", [](DSPEngine& self,
                        py::array_t<float, py::array::c_style> b0,
                        py::array_t<float, py::array::c_style> b1,
                        py::array_t<float, py::array::c_style> b2)
        {
            auto i0 = b0.request();
            auto i1 = b1.request();
            auto i2 = b2.request();

            if (i0.size != i1.size || i0.size != i2.size)
                throw std::runtime_error("Mic buffers must match");

            self.push(
                static_cast<const float*>(i0.ptr),
                static_cast<const float*>(i1.ptr),
                static_cast<const float*>(i2.ptr),
                static_cast<size_t>(i0.size)
            );
        })

        .def("ready", &DSPEngine::ready)
        .def("process", [](DSPEngine& self) {
            auto r = self.process();
            return py::make_tuple(r.angle, r.confident);
        }); 
}
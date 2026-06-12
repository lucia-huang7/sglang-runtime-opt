#include <torch/extension.h>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <c10/cuda/CUDAGuard.h>

namespace py = pybind11;

int64_t pack_lists(torch::Tensor destination, py::sequence groups) {
    TORCH_CHECK(destination.device().is_cpu() && destination.scalar_type() == at::kLong &&
                destination.is_contiguous(), "Expected a contiguous CPU int64 destination");
    int64_t count = 0;
    for (auto group : groups) count += py::len(group);
    TORCH_CHECK(count <= destination.numel(), "Integer staging capacity exceeded");
    auto* output = destination.data_ptr<int64_t>();
    int64_t offset = 0;
    for (auto group : groups) {
        for (auto value : py::reinterpret_borrow<py::sequence>(group)) {
            const auto number = PyLong_AsLongLong(value.ptr());
            if (PyErr_Occurred()) throw py::error_already_set();
            output[offset++] = number;
        }
    }
    return count;
}

std::vector<torch::Tensor> pack_and_upload(torch::Tensor host, py::sequence groups,
                                           int64_t device_index) {
    TORCH_CHECK(host.device().is_cpu() && host.scalar_type() == at::kLong &&
                host.is_contiguous() && host.is_pinned(), "Upload staging must be pinned CPU int64");
    std::vector<int64_t> lengths, offsets;
    int64_t count = 0;
    for (auto group : groups) {
        count = (count + 31) / 32 * 32;
        offsets.push_back(count);
        lengths.push_back(py::len(group));
        count += lengths.back();
    }
    TORCH_CHECK(count <= host.numel(), "Aligned staging capacity exceeded");
    auto* pointer = host.data_ptr<int64_t>();
    int64_t previous_end = 0;
    size_t index = 0;
    for (auto group : groups) {
        const auto offset = offsets[index];
        std::fill(pointer + previous_end, pointer + offset, 0);
        int64_t position = offset;
        for (auto value : py::reinterpret_borrow<py::sequence>(group)) {
            const auto number = PyLong_AsLongLong(value.ptr());
            if (PyErr_Occurred()) throw py::error_already_set();
            pointer[position++] = number;
        }
        previous_end = position;
        ++index;
    }
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_index));
    auto output = host.narrow(0, 0, count).to(torch::Device(torch::kCUDA, device_index), true);
    std::vector<torch::Tensor> result;
    for (size_t i = 0; i < lengths.size(); ++i)
        result.push_back(output.narrow(0, offsets[i], lengths[i]));
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_lists", &pack_lists, "Pack CPU integer groups into caller-owned staging");
    m.def("pack_and_upload", &pack_and_upload, "Pack and upload integer metadata on the current stream");
}

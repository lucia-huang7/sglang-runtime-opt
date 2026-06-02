#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <array>
#include <unordered_set>
#include <vector>

using Result = std::tuple<torch::Tensor, int64_t, int64_t>;

Result prepare(torch::Tensor table, const std::vector<int64_t>& rows,
               const std::vector<torch::Tensor>& sources, int64_t minimum) {
    auto invalid = []() -> Result {
        return {torch::empty({0}, torch::dtype(torch::kInt64)), -1, 0};
    };
    if (!table.is_cuda() || table.dim() != 2 || table.scalar_type() != at::kInt ||
        table.stride(1) != 1 || table.stride(0) < table.size(1) || rows.size() != sources.size()) return invalid();
    c10::cuda::CUDAGuard guard(table.device());
    auto stream = at::cuda::getCurrentCUDAStream(table.device().index());
    std::unordered_set<int64_t> seen;
    std::vector<std::array<int64_t, 4>> records;
    records.reserve(sources.size());
    int64_t maximum = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& source = sources[i];
        if (rows[i] < 0 || rows[i] >= table.size(0) || !seen.insert(rows[i]).second ||
            source.dim() != 1 || source.device() != table.device() ||
            (source.scalar_type() != at::kInt && source.scalar_type() != at::kLong) ||
            !source.is_contiguous() || source.numel() > table.size(1) ||
            source.storage().data_ptr().get() == table.storage().data_ptr().get()) return invalid();
        if (!source.numel()) continue;
        maximum = std::max(maximum, source.numel());
        records.push_back({reinterpret_cast<int64_t>(source.data_ptr()), rows[i], source.numel(),
                           source.scalar_type() == at::kLong ? 1 : 0});
    }
    if (static_cast<int64_t>(records.size()) < minimum) return invalid();
    if (records.empty()) return {torch::empty({0}, table.options().dtype(torch::kInt64)), 0, 0};
    auto host = torch::empty({static_cast<int64_t>(records.size()), 4},
                            torch::dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true));
    auto* pointer = host.data_ptr<int64_t>();
    for (const auto& record : records) {
        for (int j = 0; j < 4; ++j) *pointer++ = record[j];
    }
    for (const auto& source : sources) {
        if (source.numel()) c10::cuda::CUDACachingAllocator::recordStream(source.storage().data_ptr(), stream);
    }
    auto descriptors = host.to(table.device(), true);
    return {descriptors, static_cast<int64_t>(records.size()), maximum};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare", &prepare, "Validate and stage cached-prefix descriptors");
}

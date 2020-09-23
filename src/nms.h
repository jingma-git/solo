#pragma once
#include <torch/torch.h>

namespace F = torch::nn::functional;
using namespace torch::indexing;
inline at::Tensor point_nms(const at::Tensor &heat, int kernel = 2)
{
    auto hmax = F::max_pool2d(heat, F::MaxPool2dFuncOptions(2).stride(1).padding(1));
    auto keep = (hmax.index({Slice(), Slice(), Slice(None, -1), Slice(None, -1)}) == heat).toType(at::kFloat);
    return heat * keep;
}
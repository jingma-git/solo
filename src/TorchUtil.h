#pragma once
#include <torch/torch.h>

namespace TorchUtil
{
    inline at::Tensor bbox_area(const at::Tensor &bbox)
    {
        auto x0 = bbox.narrow(1, 0, 1);
        auto y0 = bbox.narrow(1, 1, 1);
        auto x1 = bbox.narrow(1, 2, 1);
        auto y1 = bbox.narrow(1, 3, 1);
        return torch::sqrt((x1 - x0) * (y1 - y0));
    }

    inline at::Tensor center_of_mass(const at::Tensor &binary_mask)
    {
        auto idxs = (binary_mask > 0.0).nonzero();
        auto cent_x = idxs.narrow(1, 0, 1).sum().toType(at::kFloat) / (double)idxs.size(0);
        auto cent_y = idxs.narrow(1, 1, 1).sum().toType(at::kFloat) / (double)idxs.size(0);
        return torch::stack({cent_x, cent_y});
    }
} // namespace TorchUtil
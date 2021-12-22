#pragma once
#include <torch/torch.h>

namespace F = torch::nn::functional;

// alpha * (1-pt)**gamma * log(pred)
// gamma: give 'less-focus' to well-classified samples
// alpha: give 'less-focus' to negative samples
// This method is only for debugging
// return per sample focal_loss, will apply sigmoid to prediction first
inline at::Tensor py_sigmoid_focal_loss(const at::Tensor &pred,
                                        const at::Tensor &target,
                                        double gamma = 2.0,
                                        double alpha = 0.25)
{
    auto pred_sigmoid = pred.sigmoid();

    // target = 0: pt=pred_sigmoid, focal_weight= 0.75 * (pred_sigmoid)**2
    // target = 1: pt=1-pred_sigmoid, focal_weight=0.25 * (1-pred_sigmoid)**2
    auto pt = (1.0 - pred_sigmoid) * target + pred_sigmoid * (1.0 - target);
    auto focal_weight = (alpha * target + (1.0 - alpha) * (1 - target)) * pt.pow(gamma);
    auto loss = F::binary_cross_entropy_with_logits(pred, target, F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone));
    loss = loss * focal_weight;
    return loss;
}

// return per-mask dice loss
inline at::Tensor dice_loss(const at::Tensor &pred, const at::Tensor &target)
{
    auto input = pred.contiguous().view({pred.size(0), -1});
    auto gt = target.contiguous().view({target.size(0), -1});

    auto a = torch::sum(input * gt, 1);
    auto b = torch::sum(input * input, 1) + 0.001;
    auto c = torch::sum(gt * gt, 1) + 0.001;
    auto d = (2 * a) / (b + c);
    return 1.0 - d;
}

at::Tensor reduced_dice_loss(const at::Tensor &pred, const at::Tensor &target, std::string reduction = "mean");

at::Tensor sigmoid_focal_loss(const at::Tensor &pred,
                              const at::Tensor &target,
                              int avg_factor,
                              std::string reduction = "mean",
                              double gamma = 2.0,
                              double alpha = 0.25);

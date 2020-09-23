#pragma once

#include <torch/torch.h>

#ifdef WITH_CUDA

at::Tensor SigmoidFocalLoss_forward_cuda(const at::Tensor &logits,
                                         const at::Tensor &targets,
                                         const int num_classes,
                                         const float gamma, const float alpha);

at::Tensor SigmoidFocalLoss_forward(const at::Tensor &logits,
                                    const at::Tensor &targets,
                                    const int num_classes, const float gamma,
                                    const float alpha);

at::Tensor SigmoidFocalLoss_backward_cuda(const at::Tensor &logits,
                                          const at::Tensor &targets,
                                          const at::Tensor &d_losses,
                                          const int num_classes,
                                          const float gamma, const float alpha);

at::Tensor SigmoidFocalLoss_backward(const at::Tensor &logits,
                                     const at::Tensor &targets,
                                     const at::Tensor &d_losses,
                                     const int num_classes, const float gamma,
                                     const float alpha);

class SigmoidFocalLossFunction : public torch::autograd::Function<SigmoidFocalLossFunction>
{
public:
    // bias is an optional argument
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor input, torch::Tensor target, double gamma = 2.0, double alpha = 0.25);

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grad_outputs);
};
#endif
#include "sigmoid_focal_loss.h"

#ifdef WITH_CUDA

at::Tensor SigmoidFocalLoss_forward(const at::Tensor &logits,
                                    const at::Tensor &targets,
                                    const int num_classes, const float gamma,
                                    const float alpha)
{
    if (logits.type().is_cuda())
    {
        return SigmoidFocalLoss_forward_cuda(logits, targets, num_classes, gamma,
                                             alpha);
    }
    AT_ERROR("SigmoidFocalLoss is not implemented on the CPU");
}

at::Tensor SigmoidFocalLoss_backward(const at::Tensor &logits,
                                     const at::Tensor &targets,
                                     const at::Tensor &d_losses,
                                     const int num_classes, const float gamma,
                                     const float alpha)
{
    if (logits.type().is_cuda())
    {
        return SigmoidFocalLoss_backward_cuda(logits, targets, d_losses,
                                              num_classes, gamma, alpha);
    }
    AT_ERROR("SigmoidFocalLoss is not implemented on the CPU");
}

torch::Tensor SigmoidFocalLossFunction::forward(torch::autograd::AutogradContext *ctx, torch::Tensor input, torch::Tensor target, double gamma, double alpha)
{
    ctx->save_for_backward({input, target});
    auto num_classes = input.size(1);
    ctx->saved_data["num_classes"] = num_classes;
    ctx->saved_data["gamma"] = gamma;
    ctx->saved_data["alpha"] = alpha;

    auto loss =
        SigmoidFocalLoss_forward(input, target, num_classes, gamma,
                                 alpha);

    return loss;
}

torch::autograd::tensor_list SigmoidFocalLossFunction::backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list d_loss)
{
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto target = saved[1];

    auto num_classes = ctx->saved_data["num_classes"].toInt();
    auto gamma = ctx->saved_data["gamma"].toDouble();
    auto alpha = ctx->saved_data["alpha"].toDouble();
    auto d_input = SigmoidFocalLoss_backward(input, target, d_loss[0],
                                             num_classes, gamma, alpha);
    return {d_input,
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable()};
}
#endif
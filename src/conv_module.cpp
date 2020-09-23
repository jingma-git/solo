#include "conv_module.h"
#include "config.h"
#include <iostream>
using namespace std;

namespace nn = torch::nn;
namespace F = torch::nn::functional;

ConvModuleImpl::ConvModuleImpl() {}

ConvModuleImpl::ConvModuleImpl(int in_channels, int out_channels, int kernel_size,
                               int stride, int padding, std::string norm_, std::string activation_) : norm(norm_), activation(activation_)
{
    conv = nn::Conv2d(nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(stride).padding(padding));
    // nn::init::kaiming_normal_(conv->weight, 0, torch::kFanOut, torch::kReLU);
    register_module("conv", conv);

    if (norm == "gn")
    {
        gn = nn::GroupNorm(nn::GroupNormOptions(Cfg::num_groups, out_channels));
        register_module("norm", gn);
    }
    else if (norm == "bn")
    {
        bn = nn::BatchNorm2d(nn::BatchNorm2dOptions(out_channels));
        register_module("norm", bn);
    }
}

at::Tensor ConvModuleImpl::forward(const at::Tensor &input)
{
    auto x = conv->forward(input);
    if (norm == "gn")
    {
        x = gn->forward(x);
    }
    else if (norm == "bn")
    {
        x = bn->forward(x);
    }

    if (activation == "relu")
    {
        x = F::relu(x);
    }

    return x;
}
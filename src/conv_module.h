#pragma once
#include <torch/torch.h>
#include <string>

class ConvModuleImpl : public torch::nn::Module
{
public:
    ConvModuleImpl();
    ConvModuleImpl(int in_channels, int out_channels, int kernel_size,
                   int stride = 1, int padding = 0, std::string norm_ = "", std::string activation_ = "");

    at::Tensor forward(const at::Tensor &input);

private:
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
    torch::nn::GroupNorm gn{nullptr};

    std::string norm;
    std::string activation;
};
TORCH_MODULE(ConvModule);
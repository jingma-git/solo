#pragma once
#include "resnet.h"
#include <torch/torch.h>

struct FPNOut
{
    at::Tensor p2;
    at::Tensor p3;
    at::Tensor p4;
    at::Tensor p5;
    at::Tensor p6;
};

class FPNImpl : public torch::nn::Module
{
public:
    FPNImpl();

    std::vector<at::Tensor> forward(torch::Tensor);

private:
    ResNet bottom_up_{nullptr};

    torch::nn::Conv2d fpn_output2_{nullptr};
    torch::nn::Conv2d fpn_output3_{nullptr};
    torch::nn::Conv2d fpn_output4_{nullptr};
    torch::nn::Conv2d fpn_output5_{nullptr};
    torch::nn::MaxPool2d fpn_output6_{nullptr};

    torch::nn::Conv2d fpn_lateral2_{nullptr};
    torch::nn::Conv2d fpn_lateral3_{nullptr};
    torch::nn::Conv2d fpn_lateral4_{nullptr};
    torch::nn::Conv2d fpn_lateral5_{nullptr};
};

TORCH_MODULE(FPN);
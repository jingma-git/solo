#pragma once
#include <torch/torch.h>


// Input (N x 3 x H x W)
// Return: (N x 64 x H/4 x W/4)
class BasicStemImpl : public torch::nn::Module
{
public:
    BasicStemImpl();
    at::Tensor forward(const at::Tensor &input);

private:
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
};
TORCH_MODULE(BasicStem);

// Input (N x 256 x H x W)
// Output (N x ? x H x W)
class BottleNeckImpl : public torch::nn::Module
{
public:
    BottleNeckImpl();
    BottleNeckImpl(int64_t in_channels, int64_t out_channels, int64_t bottleneck_channels, int64_t stride = 1);
    at::Tensor forward(const at::Tensor &input);

    int64_t expansion = 4;

private:
    torch::nn::Conv2d conv1{nullptr}; // 1 x 1 conv, 64
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d conv2{nullptr}; // 3 x 3 conv, 64
    torch::nn::BatchNorm2d bn2{nullptr};
    torch::nn::Conv2d conv3{nullptr}; // 1 x 1 conv, 256
    torch::nn::BatchNorm2d bn3{nullptr};
    torch::nn::Conv2d shortcut{nullptr};
};
TORCH_MODULE(BottleNeck);

struct ResNetOut
{
    at::Tensor c2;
    at::Tensor c3;
    at::Tensor c4;
    at::Tensor c5;
};

class ResNetImpl : public torch::nn::Module
{
public:
    ResNetImpl();
    ResNetOut forward(const at::Tensor &input);

    torch::nn::Sequential make_layer(int planes, int blocks, int stride = 1);

private:
    BasicStem stem{nullptr};
    // Each resnet layer double # of filters and downsample spatially using stride 2
    torch::nn::Sequential res2{nullptr};
    torch::nn::Sequential res3{nullptr};
    torch::nn::Sequential res4{nullptr};
    torch::nn::Sequential res5{nullptr};

    std::vector<int> blocks;
    int64_t inplanes = 64;

    std::map<int, std::vector<int>> arch_settings = {{1, {2, 2, 2, 2}},
                                                     {34, {3, 4, 6, 3}},
                                                     {50, {3, 4, 6, 3}},
                                                     {101, {3, 4, 23, 3}}};
};
TORCH_MODULE(ResNet);

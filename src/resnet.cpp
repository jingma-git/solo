#include "resnet.h"
#include "config.h"

// #define DEBUG_ResNet

namespace nn = torch::nn;
namespace F = torch::nn::functional;

BasicStemImpl::BasicStemImpl()
{
    conv = nn::Conv2d(nn::Conv2dOptions(Cfg::in_channels, 64, 7).stride(2).padding(3).bias(false));
    bn = nn::BatchNorm2d(nn::BatchNorm2dOptions(64));

    for (auto param : bn->parameters())
    {
        param.requires_grad_(true);
    }

    register_module("conv", conv);
    register_module("bn", bn);
}

// Input (N x 3 x H x W)
// Return: (N x 64 x H/4 x W/4)
at::Tensor BasicStemImpl::forward(const at::Tensor &input)
{
    auto x = conv(input);
    x = bn(x);
    x = F::relu(x);
    x = F::max_pool2d(x, F::MaxPool2dFuncOptions(3).stride(2).padding(1));
    return x;
}

BottleNeckImpl::BottleNeckImpl()
{
}

BottleNeckImpl::BottleNeckImpl(int64_t in_channels, int64_t out_channels, int64_t bottleneck_channels, int64_t stride)
{
    if (in_channels != out_channels)
    {
        shortcut = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false));
        register_module("shortcut", shortcut);
    }

    conv1 = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, bottleneck_channels, 1).stride(stride).bias(false));
    conv2 = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(bottleneck_channels, bottleneck_channels, 3).stride(1).padding(1).bias(false));
    conv3 = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(bottleneck_channels, out_channels, 1).stride(1).bias(false));
    bn1 = nn::BatchNorm2d(nn::BatchNorm2dOptions(bottleneck_channels));
    bn2 = nn::BatchNorm2d(nn::BatchNorm2dOptions(bottleneck_channels));
    bn3 = nn::BatchNorm2d(nn::BatchNorm2dOptions(out_channels));
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("bn1", bn1);
    register_module("bn2", bn2);
    register_module("bn3", bn3);
}

at::Tensor BottleNeckImpl::forward(const at::Tensor &input)
{
    torch::Tensor x = conv1->forward(input);
    x = bn1(x);
    x = F::relu(x);

    x = conv2(x);
    x = bn2(x);
    x = F::relu(x);

    x = conv3(x);
    x = bn3(x);

    torch::Tensor tmp;
    if (!shortcut.is_empty())
    {
        tmp = shortcut->forward(input);
    }
    else
    {
        tmp = input;
    }

    x += tmp;
    x = F::relu(x);
    return x;
}

ResNetImpl::ResNetImpl()
{
    stem = BasicStem();
    register_module("stem", stem);

    //res2: 64-64, 64-64, 64-256
    //res3: 256-128, 128-128, 128-512
    //res4: 512-256, 256-256, 256-1024
    //res5: 1024-512, 512-512, 512-2048
    blocks = arch_settings[Cfg::resnet_arch];
    res2 = make_layer(64, blocks[0]);
    res3 = make_layer(128, blocks[1], 2);
    res4 = make_layer(256, blocks[2], 2);
    res5 = make_layer(512, blocks[3], 2);

    register_module("res2", res2);
    register_module("res3", res3);
    register_module("res4", res4);
    register_module("res5", res5);

    //ToDo: freeze stages
    //ToDo: init weights
}

ResNetOut ResNetImpl::forward(const at::Tensor &input)
{
    ResNetOut out;
    auto x = stem->forward(input);

    out.c2 = res2->forward(x);
    out.c3 = res3->forward(out.c2);
    out.c4 = res4->forward(out.c3);
    out.c5 = res5->forward(out.c4);

#ifdef DEBUG_ResNet
    std::cout << "Stem Output:" << x.sizes() << std::endl;
    std::cout << "c2:" << out.c2.sizes() << std::endl;
    std::cout << "c3:" << out.c3.sizes() << std::endl;
    std::cout << "c4:" << out.c4.sizes() << std::endl;
    std::cout << "c5:" << out.c5.sizes() << std::endl;
#endif

    return out;
}

torch::nn::Sequential ResNetImpl::make_layer(int planes, int blocks, int stride)
{
    if (planes == 64)
    {
        inplanes = 64;
    }
    else
    {
        inplanes = planes * 2;
    }

    torch::nn::Sequential layers(
        BottleNeck(inplanes, planes * 4, planes, stride));

    for (int i = 1; i < blocks; i++)
    {
        layers->push_back(BottleNeck(planes * 4, planes * 4, planes));
    }
    return layers;
}
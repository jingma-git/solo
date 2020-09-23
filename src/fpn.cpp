#include "resnet.h"
#include "fpn.h"

// #define DEBUG_FPN
// #define DEBUG_FPN_Data

namespace F = torch::nn::functional;

FPNImpl::FPNImpl()
{
    bottom_up_ = ResNet();
    register_module("bottom_up", bottom_up_);

    fpn_lateral2_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 1));
    fpn_lateral3_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 256, 1));
    fpn_lateral4_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 256, 1));
    fpn_lateral5_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(2048, 256, 1));
    register_module("fpn_lateral2", fpn_lateral2_);
    register_module("fpn_lateral3", fpn_lateral3_);
    register_module("fpn_lateral4", fpn_lateral4_);
    register_module("fpn_lateral5", fpn_lateral5_);

    fpn_output2_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1));
    fpn_output3_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1));
    fpn_output4_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1));
    fpn_output5_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1));
    fpn_output6_ = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(1).stride(2));
    register_module("fpn_output2", fpn_output2_);
    register_module("fpn_output3", fpn_output3_);
    register_module("fpn_output4", fpn_output4_);
    register_module("fpn_output5", fpn_output5_);
    register_module("fpn_output6", fpn_output6_);
}

std::vector<at::Tensor> FPNImpl::forward(torch::Tensor image)
{
    torch::Tensor c2, c3, c4, c5;
    ResNetOut res_out = bottom_up_->forward(image);
    c2 = res_out.c2;
    c3 = res_out.c3;
    c4 = res_out.c4;
    c5 = res_out.c5;

    torch::Tensor feature5 = fpn_lateral5_->forward(c5);
    torch::Tensor out5 = fpn_output5_->forward(feature5);

    torch::Tensor feature4 = F::interpolate(
        feature5,
        F::InterpolateFuncOptions().scale_factor(std::vector<double>({2.0, 2.0})).mode(torch::kNearest).recompute_scale_factor(true));
    feature4 += fpn_lateral4_->forward(c4); // Should I change to Bilinear
    torch::Tensor out4 = fpn_output4_->forward(feature4);

    torch::Tensor feature3 = F::interpolate(
        feature4,
        F::InterpolateFuncOptions().scale_factor(std::vector<double>({2.0, 2.0})).mode(torch::kNearest).recompute_scale_factor(true));
    feature3 += fpn_lateral3_->forward(c3);
    torch::Tensor out3 = fpn_output3_->forward(feature3);

    torch::Tensor feature2 = F::interpolate(
        feature3,
        F::InterpolateFuncOptions().scale_factor(std::vector<double>({2.0, 2.0})).mode(torch::kNearest).recompute_scale_factor(true));
    feature2 += fpn_lateral2_->forward(c2);
    torch::Tensor out2 = fpn_output2_->forward(feature2);

    torch::Tensor out6 = fpn_output6_->forward(out5);

#ifdef DEBUG_FPN
    std::cout << "p2: " << out2.sizes() << std::endl;
    std::cout << "p3: " << out3.sizes() << std::endl;
    std::cout << "p4: " << out4.sizes() << std::endl;
    std::cout << "p5: " << out5.sizes() << std::endl;
    std::cout << "p6: " << out6.sizes() << std::endl;
#endif

#ifdef DEBUG_FPN_Data
    using namespace torch::indexing;
    std::cout << "p2: " << out2.index({0, 0, Slice(0, None, 8), Slice(0, None, 8)}) << std::endl;
    std::cout << "p3: " << out3.index({0, 0, Slice(0, None, 4), Slice(0, None, 4)}) << std::endl;
    std::cout << "p4: " << out4.index({0, 0, Slice(0, None, 2), Slice(0, None, 2)}) << std::endl;
    std::cout << "p5: " << out5.index({0, 0}) << std::endl;
    std::cout << "p6: " << out6.index({0, 0}) << std::endl;
#endif

    // FPNOut out;
    // out.p2 = out2;
    // out.p3 = out3;
    // out.p4 = out4;
    // out.p5 = out.p5;
    // out.p6 = out.p6;
    return {out2, out3, out4, out5, out6};
}
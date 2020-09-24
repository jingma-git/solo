#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include "conv_module.h"

struct SoloOut
{
    std::vector<at::Tensor> ins_preds;
    std::vector<at::Tensor> cate_preds;
};

class SOLOHeadImpl : public torch::nn::Module
{
public:
    SOLOHeadImpl();

    SoloOut forward(const std::vector<at::Tensor> &feats);

private:
    std::vector<at::Tensor> split_feats(const std::vector<at::Tensor> &feats);

private:
    int in_channels = 256;
    int seg_feat_channels = 256;

    std::vector<ConvModule> ins_convs; //instance branch
    std::vector<torch::nn::Conv2d> solo_ins_list;

    std::vector<ConvModule> cate_convs; // category branch
    torch::nn::Conv2d solo_cate{nullptr};
};
TORCH_MODULE(SOLOHead);
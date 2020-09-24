#include "config.h"
#include "conv_module.h"
#include "solo_head.h"
#include "nms.h"
#include <iostream>
#include <omp.h>

namespace nn = torch::nn;
namespace F = torch::nn::functional;
using namespace std;

SOLOHeadImpl::SOLOHeadImpl()
{
    for (int i = 0; i < Cfg::stacked_convs; i++)
    {
        int chn = (i == 0) ? in_channels + 2 : seg_feat_channels;
        // int in_channels, int out_channels,
        // int kernel_size, int stride = 1, int padding = 0,
        ins_convs.push_back(ConvModule(chn, seg_feat_channels, 3, 1, 1, "gn", "relu"));
        chn = (i == 0) ? in_channels : seg_feat_channels;
        cate_convs.push_back(ConvModule(chn, seg_feat_channels, 3, 1, 1, "gn", "relu"));
        register_module("ins_conv" + to_string(i), ins_convs[i]);
        register_module("cate_conv" + to_string(i), cate_convs[i]);
    }

    for (size_t i = 0; i < Cfg::num_grids.size(); i++)
    {
        int seg_num_grid = Cfg::num_grids[i];
        solo_ins_list.push_back(nn::Conv2d(nn::Conv2dOptions(seg_feat_channels, seg_num_grid * seg_num_grid, 1)));
        register_module("ins_pred" + to_string(i), solo_ins_list[i]);
    }

    solo_cate = nn::Conv2d(nn::Conv2dOptions(seg_feat_channels, Cfg::num_classes - 1, 3).padding(1));
    register_module("cate_pred", solo_cate);
}

SoloOut SOLOHeadImpl::forward(const std::vector<at::Tensor> &feats)
{
    int64_t N = feats[0].size(0); //batch_size
    std::vector<at::Tensor> new_feats = split_feats(feats);

    SoloOut out;
    out.ins_preds.resize(new_feats.size());
    out.cate_preds.resize(new_feats.size());

    // #pragma omp parallel for
    for (size_t i = 0; i < new_feats.size(); i++)
    {
        // Forward features for each level
        at::Tensor ins_feat = new_feats[i];
        at::Tensor cate_feat = new_feats[i];
        int seg_num_grid = Cfg::num_grids[i];

        // instance branch
        auto x_range = torch::linspace(-1, 1, ins_feat.size(3));
        auto y_range = torch::linspace(-1, 1, ins_feat.size(2));
        auto grid = torch::meshgrid({y_range, x_range});
        auto y = grid[0], x = grid[1];
        y = y.expand({N, 1, -1, -1});
        x = x.expand({N, 1, -1, -1});
        auto coord_feat = torch::cat({x, y}, 1).to(device);
        ins_feat = torch::cat({ins_feat, coord_feat}, 1);

        for (size_t l = 0; l < ins_convs.size(); l++)
        {
            ins_feat = ins_convs[l]->forward(ins_feat);
        }

        ins_feat = F::interpolate(ins_feat, F::InterpolateFuncOptions().scale_factor(std::vector<double>({2.0, 2.0})).mode(torch::kBilinear).align_corners(false).recompute_scale_factor(true));
        auto ins_pred = solo_ins_list[i]->forward(ins_feat);

        // category branch
        for (size_t l = 0; l < cate_convs.size(); l++)
        {
            if (l == 0)
            {
                cate_feat = F::interpolate(cate_feat, F::InterpolateFuncOptions().size(std::vector<int64_t>({seg_num_grid, seg_num_grid})).mode(torch::kBilinear).align_corners(false));
            }
            cate_feat = cate_convs[l]->forward(cate_feat);
        }
        auto cate_pred = solo_cate->forward(cate_feat);

        if (false)
        {
            cout << i << " ins_feat=" << ins_feat.sizes() << " y=" << y.sizes() << " x=" << x.sizes() << " coord=" << coord_feat.sizes();
            cout << " ins_pred=" << ins_pred.sizes() << " cate_pred=" << cate_pred.sizes() << endl;
        }
        out.ins_preds[i] = ins_pred;
        out.cate_preds[i] = cate_pred;
    }

    return out;
}

std::vector<at::Tensor> SOLOHeadImpl::split_feats(const std::vector<at::Tensor> &feats)
{
    return {
        F::interpolate(feats[0], F::InterpolateFuncOptions().scale_factor(std::vector<double>({0.5, 0.5})).mode(torch::kBilinear).align_corners(false).recompute_scale_factor(true)),
        feats[1],
        feats[2],
        feats[3],
        F::interpolate(feats[4], F::InterpolateFuncOptions().scale_factor(std::vector<double>({2.0, 2.0})).mode(torch::kBilinear).align_corners(false).recompute_scale_factor(true))};
}
#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include "dataset.h"
#include "fpn.h"
#include "solo_head.h"

struct SoloLoss
{
    at::Tensor ins_loss;  // instance loss
    at::Tensor cate_loss; // category loss
};

struct SoloPred
{
    at::Tensor cate_scores; //category scores
    at::Tensor cate_labels;
    at::Tensor seg_masks;
};

class SOLOImpl : public torch::nn::Module
{
public:
    SOLOImpl();

    SoloOut forward(const at::Tensor &input);
    SoloLoss loss(const SoloOut &pred, std::vector<Sample> &sample);

    // Input: processed imags
    SoloPred predict(const at::Tensor &input);

    // Input: raw prediction from solo_head
    // Output: nms processed category labels, scores, and instance mask
    bool post_process(SoloOut &head_out, SoloPred &solo_pred);
    void visualize_pred(SoloPred &pred, std::string save_dir);
    void visualize_input(const at::Tensor &image, std::string save_dir);

private:
private:
    FPN fpn{nullptr};
    SOLOHead head{nullptr};

    std::vector<int> strides = {8, 8, 16, 32, 32};

    // p_scale = 1.0 / (strides[i]/2)
    std::vector<double> p_scales = {1 / 4.0, 1 / 4.0, 1 / 8.0, 1 / 16.0, 1 / 16.0}; // scale to map the original mask to corresponding pyramid level
    double radius_scale = 0.2;
};
TORCH_MODULE(SOLO);
#pragma once
#include <torch/torch.h>
#include <string>
#include <iostream>
#include <map>

#ifdef WITH_CUDA
const torch::Device device(torch::kCUDA);
#else
const torch::Device device(torch::kCPU);
#endif

struct Cfg
{
    static std::vector<std::string> class_names;

    //************************Model**************************
    //
    static bool pretrain;
    // ResNet
    static int resnet_arch;
    static int in_channels;
    // SOLOHead
    static int num_classes;
    static std::vector<int> num_grids;
    static std::vector<std::vector<int>> scale_ranges;
    static int stacked_convs;
    static int num_groups;
    static std::vector<int> strides;
    // Loss
    static std::string loss_cat;
    static std::string loss_ins;
    static double lambda_cat;
    static double lambda_ins;
    // Multi-view
    static int img_size;
    static int input_size; // the input image size for the network
    static double img_scale;
    static std::vector<std::string> input_view_names;
    static std::vector<std::string> target_view_names;
    static std::string input_views;
    static int num_target_views;
    static int num_views;
    static std::string style_ids;

    //************************Data**************************
    static std::string data_dir;
    static std::string output_dir;

    //************************Train**************************
    static int epoch;
    static int val_epoch;
    static float lr;
    static float weight_decay;
    static int batch_size;

    //************************Predict**************************
    static double score_thr;
    static double mask_thr;
    static double update_thr;
    static int nms_pre;
    static int max_per_img;

    static int get_cls_idx(std::string cls_name);
};
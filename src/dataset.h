#pragma once

#include <torch/torch.h>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

namespace fs = boost::filesystem;

struct TargetData
{
    std::string dn_path;
    std::map<std::string, std::vector<double>> contours;
};

struct RawData
{
    // record front-view image path if Cfg::input_views=="FS" or Cfg::input_views=="F"
    // record side-view image path if Cfg::input_views=="S"
    // record arbitrary-view image path if Cfg::input_views=="A"
    std::string img_path;

    // record side-view image path if Cfg::input_views=="FS"
    std::string img_path_side;

    // record all contour depth info in memory
    // std::map<std::string, std::vector<double>> contours; // Key: cls_name, Val: contour

    // depth-normal paths for corresponding input image/images
    std::vector<TargetData> targets;
    char styleID;
};

struct Input
{
    at::Tensor image;
};

struct Target
{
    at::Tensor gt_classes;
    at::Tensor gt_bboxs;
    at::Tensor gt_masks;
};

using Sample = torch::data::Example<Input, Target>;

class SoloDataset : public torch::data::Dataset<SoloDataset, Sample>
{
public:
    SoloDataset(std::string filename);

    Sample get(size_t index) override;
    torch::optional<size_t> size() const override;

private:
    bool read_shape_list();
    void build_raw_data();
    bool parseSketch(const std::string imgName, cv::Mat &sketch);

    std::vector<std::string> shape_list;
    std::vector<RawData> raw_datas;

    std::string filename;
    fs::path data_dir;
    fs::path file_path;
};
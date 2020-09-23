#include "dataset.h"

#include <iostream>
#include <fstream>
#include <CvUtil.h>
#include <JsonUtil.h>
#include <TimeUtil.h>

#include "dataset.h"
#include "config.h"
#include "ImgUtil.h"

using namespace std;

SoloDataset::SoloDataset(std::string filename) : filename(filename)
{
    data_dir = fs::path(Cfg::data_dir);
    file_path = data_dir / filename;
    read_shape_list();

    build_raw_data();
}

bool SoloDataset::read_shape_list()
{
    if (!fs::exists(file_path))
    {
        cerr << __FILE__ << " " << __LINE__ << ":" << file_path << " does not exists" << endl;
        return false;
    }
    ifstream file(file_path.c_str());
    string line;
    while (getline(file, line))
    {
        shape_list.push_back(line);
    }

    return true;
}

void SoloDataset::build_raw_data()
{
    auto add_target = [&](std::string &shape_name, RawData &raw_data) {
        for (int view_id = 0; view_id < Cfg::num_target_views; view_id++)
        {
            TargetData target;
            char target_name[50];
            sprintf(target_name, "T_%d_d.json", view_id);
            target.dn_path = (data_dir / "color" / shape_name / target_name).c_str();
            string contour_path = (data_dir / "sketch" / shape_name / ("T_" + to_string(view_id) + "_contour.json")).c_str();
            JsonUtil::load_annos(contour_path, target.contours);
            for (auto &kv : target.contours)
            {
                auto &contour = kv.second;
                for (size_t k = 0; k < contour.size(); k += 3)
                {
                    int x = int(contour[k] * Cfg::img_scale + 0.5);
                    int y = int(contour[k + 1] * Cfg::img_scale + 0.5);
                    if (x < 0)
                        x = 0;
                    if (x > (Cfg::input_size - 1))
                        x = Cfg::input_size - 1;
                    if (y < 0)
                        y = 0;
                    if (y > (Cfg::input_size - 1))
                        y = Cfg::input_size - 1;
                    contour[k] = x;
                    contour[k + 1] = contour[k + 1] * Cfg::img_scale;
                }
            }
            raw_data.targets.push_back(target);
        }
    };

    for (size_t i = 0; i < shape_list.size(); i++)
    {
        string &shape_name = shape_list[i];
        for (size_t j = 0; j < Cfg::style_ids.size(); j++)
        {
            if (Cfg::input_views == "FS")
            {
                for (int k = 0; k < 10; k++) // View Angle
                {
                    RawData raw_data;
                    char styleID = Cfg::style_ids[j];
                    raw_data.styleID = styleID;

                    char img_name_front[50], img_name_side[50];
                    sprintf(img_name_front, "F_%d_sketch%c.jpg", k, styleID);
                    sprintf(img_name_side, "S_%d_sketch%c.jpg", k, styleID);
                    raw_data.img_path = (data_dir / "sketch" / shape_name / img_name_front).c_str();
                    raw_data.img_path_side = (data_dir / "sketch" / shape_name / img_name_side).c_str();

                    // TargetData target_front;
                    // target_front.dn_path = (data_dir / "color" / shape_name / ("F_" + to_string(k) + "_d.json")).c_str();
                    // string contour_front = (data_dir / "sketch" / shape_name / ("F_" + to_string(k) + "_contour.json")).c_str();
                    // JsonUtil::load_annos(contour_front, target_front.contours);

                    // TargetData target_side;
                    // target_front.dn_path = (data_dir / "color" / shape_name / ("S_" + to_string(k) + "_d.json")).c_str();
                    // string contour_side = (data_dir / "sketch" / shape_name / ("S_" + to_string(k) + "_contour.json")).c_str();
                    // JsonUtil::load_annos(contour_side, target_side.contours);

                    // raw_data.targets.push_back(target_front);
                    // raw_data.targets.push_back(target_side);

                    add_target(shape_name, raw_data);
                    raw_datas.push_back(raw_data);
                }
            }
            else if (Cfg::input_views == "F")
            {
                for (int k = 0; k < 10; k++) // View Angle, for each view there are 10 sub-views with slightly different angles
                {
                    RawData raw_data;
                    char styleID = Cfg::style_ids[j];
                    raw_data.styleID = styleID;

                    char img_name_front[50];
                    sprintf(img_name_front, "F_%d_sketch%c.jpg", k, styleID);
                    raw_data.img_path = (data_dir / "sketch" / shape_name / img_name_front).c_str();

                    // TargetData target_front;
                    // target_front.dn_path = (data_dir / "color" / shape_name / ("F_" + to_string(k) + "_d.json")).c_str();
                    // string contour_front = (data_dir / "sketch" / shape_name / ("F_" + to_string(k) + "_contour.json")).c_str();
                    // JsonUtil::load_annos(contour_front, target_front.contours);

                    // raw_data.targets.push_back(target_front);

                    add_target(shape_name, raw_data);
                    raw_datas.push_back(raw_data);
                }
            }
            else if (Cfg::input_views == "S")
            {
                for (int k = 0; k < 10; k++) // View Angle
                {
                    RawData raw_data;
                    char styleID = Cfg::style_ids[j];
                    raw_data.styleID = styleID;

                    char img_name_side[50];
                    sprintf(img_name_side, "S_%d_sketch%c.jpg", k, styleID);
                    raw_data.img_path_side = (data_dir / "sketch" / shape_name / img_name_side).c_str();

                    // TargetData target_side;
                    // string contour_side = (data_dir / "sketch" / shape_name / ("S_" + to_string(k) + "_contour.json")).c_str();
                    // JsonUtil::load_annos(contour_side, target_side.contours);

                    // raw_data.targets.push_back(target_side);
                    add_target(shape_name, raw_data);
                    raw_datas.push_back(raw_data);
                }
            }
        }
    }
}

Sample SoloDataset::get(size_t index)
{
    using namespace cv;
    const RawData &raw_data = raw_datas[index];
    Sample result;

    TimeUtil::Timer timer;
    cv::Mat img, img_side;
    if (Cfg::input_views == "FS")
    {
        if (!parseSketch(raw_data.img_path, img))
        {
            cerr << __FILE__ << " " << __LINE__ << ": " << raw_data.img_path << " doesn't exist" << endl;
            exit(-1);
        }

        if (!parseSketch(raw_data.img_path_side, img_side))
        {
            cerr << __FILE__ << " " << __LINE__ << ": " << raw_data.img_path_side << " doesn't exist" << endl;
            exit(-1);
        }
        cv::resize(img, img, cv::Size(Cfg::input_size, Cfg::input_size));
        cv::resize(img_side, img_side, cv::Size(Cfg::input_size, Cfg::input_size));
        img = ImgUtil::normalize_img(img);
        img_side = ImgUtil::normalize_img(img_side);
        auto img_tensor = ImgUtil::CvImageToTensor(img);
        auto img_side_tensor = ImgUtil::CvImageToTensor(img_side);
        result.data.image = torch::stack({img_tensor, img_side_tensor});

        if (false)
        {
            cv::Mat un_img = ImgUtil::TensorToCvMat(img_tensor);
            un_img = ImgUtil::unnormalize_img(un_img);
            un_img.convertTo(un_img, CV_8U);
            cv::imshow("img_front_orig", img);
            cv::imshow("img_front", un_img);
            cv::Mat un_img_side = ImgUtil::TensorToCvMat(img_side_tensor);
            un_img_side = ImgUtil::unnormalize_img(un_img_side);
            cv::imshow("img_side_orig", img_side);
            un_img_side.convertTo(un_img, CV_8U);
            cv::imshow("img_side", un_img_side);
            cv::waitKey(-1);
        }
    }
    else
    {
        if (!parseSketch(raw_data.img_path, img))
        {
            cerr << __FILE__ << " " << __LINE__ << ": " << raw_data.img_path << " doesn't exist" << endl;
            exit(-1);
        }
        cv::resize(img, img, cv::Size(Cfg::input_size, Cfg::input_size));
        img = ImgUtil::normalize_img(img);
        auto img_tensor = ImgUtil::CvImageToTensor(img);
        result.data.image = torch::stack({img_tensor});

        if (false)
        {
            cv::Mat un_img = ImgUtil::TensorToCvMat(img_tensor);
            un_img = ImgUtil::unnormalize_img(un_img);
            un_img.convertTo(un_img, CV_8U);
            cv::imshow("img_front_orig", img);
            cv::imshow("img", un_img);
            cv::imwrite("img_front.jpg", un_img);
            cv::waitKey(-1);
        }
    }
    // arranged in view0{cls0--clsN}--viewN{cls0--clsN} fashion
    std::vector<torch::Tensor> bbox_vec;
    std::vector<torch::Tensor> mask_vec;
    std::vector<torch::Tensor> depth_vec;
    std::vector<torch::Tensor> contour_vec;
    std::vector<int> classes;
    for (size_t i = 0; i < raw_data.targets.size(); i++) // # views
    {
        auto &contours = raw_data.targets[i].contours;

        FileStorage file_storage(raw_data.targets[i].dn_path, FileStorage::READ);
        for (auto &kv : contours)
        {
            std::string cls_name = kv.first;
            int cls_idx = Cfg::get_cls_idx(cls_name);
            if (cls_idx == -1)
                continue;

            classes.push_back(cls_idx);

            const std::vector<double> &contour = kv.second;
            torch::Tensor contour_tensor = torch::tensor(contour, at::kFloat);
            contour_vec.push_back(contour_tensor);
            at::Tensor contour_tmp = contour_tensor.view({-1, 3});
            auto x_min = contour_tmp.narrow(1, 0, 1).min();
            auto x_max = contour_tmp.narrow(1, 0, 1).max();
            auto y_min = contour_tmp.narrow(1, 1, 1).min();
            auto y_max = contour_tmp.narrow(1, 1, 1).max();
            auto bbox = torch::stack({x_min, y_min, x_max, y_max});

            cv::Mat depth;
            file_storage[cls_name.c_str()] >> depth;

            depth.convertTo(depth, CV_32FC1);
            cv::patchNaNs(depth, 0.0);
            cv::resize(depth, depth, cv::Size(Cfg::input_size, Cfg::input_size));

            auto depth_tensor = ImgUtil::CvImageToTensor(depth);
            if (false)
            {
                cv::Mat tmp = ImgUtil::TensorToCvMat(depth_tensor);
                tmp = tmp * 65535.0;
                tmp.convertTo(tmp, CV_16UC1);
                cv::imshow("depth", tmp);
            }

            cv::Mat mask = (depth > 0) / 255.0;
            auto mask_tensor = ImgUtil::CvImageToTensor(mask);
            if (false)
            {
                cv::Mat tmp = ImgUtil::TensorToCvMat(mask_tensor);
                tmp = tmp * 255.0;
                tmp.convertTo(tmp, CV_8UC1);
                int x0 = x_min.detach().cpu().item().toDouble();
                int y0 = y_min.detach().cpu().item().toDouble();
                int x1 = x_max.detach().cpu().item().toDouble();
                int y1 = y_max.detach().cpu().item().toDouble();
                cv::rectangle(tmp, cv::Rect(x0, y0, x1 - x0, y1 - y0), 255, 3);
                cv::imshow("mask", tmp);
                // cout << "mask_tensor " << mask_tensor << endl;
                cv::waitKey(-1);
            }

            bbox_vec.push_back(bbox);
            mask_vec.push_back(mask_tensor);
            depth_vec.push_back(depth_tensor);
        }
    }

    // ground truth batch is arranged by order from view0(cls0, ..., clsM)-->viewN(cls0, ..., clsM)
    result.target.gt_classes = torch::tensor(classes, at::kLong);
    result.target.gt_bboxs = torch::stack(bbox_vec);
    result.target.gt_masks = torch::stack(mask_vec);
    // result.target.gt_depths = torch::stack(depth_vec); // (#views x #cls) x H x W
    // result.target.contours = contour_vec;

    if (false)
    {
        cout << ">>>>>>>>>>>>>>>>>>>>>Prepare Targets takes " << timer.elapsed() << " s\n";
        cout << raw_data.img_path << endl;
        // cout << raw_data.img_path << ":" << CvUtil::type2str(img.type()) << "->" << result.data.image.sizes() << "-" << result.data.image.type() << endl;
        cout << "bbox: " << result.target.gt_bboxs.sizes() << endl;
        cout << "cls: " << result.target.gt_classes.sizes() << endl;
        cout << "mask: " << result.target.gt_masks.sizes() << endl;
        // cout << "depth: " << result.target.gt_depths.sizes() << endl;
    }
    return result;
}

torch::optional<size_t> SoloDataset::size() const
{
    return raw_datas.size();
}

bool SoloDataset::parseSketch(const std::string imgName, cv::Mat &sketch)
{
    if (!fs::exists(imgName))
        return false;

    sketch = cv::imread(imgName, cv::IMREAD_UNCHANGED);

    if (sketch.channels() == 3)
    {
        cv::cvtColor(sketch, sketch, cv::COLOR_BGR2GRAY);
    }
    return true;
}

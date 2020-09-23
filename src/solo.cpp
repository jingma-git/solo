#include <iostream>
#include "solo.h"
#include "config.h"
#include "TorchUtil.h"
#include "loss.h"
#include <ImgUtil.h>

using namespace std;
using namespace torch::indexing;
namespace F = torch::nn::functional;

SOLOImpl::SOLOImpl()
{
    fpn = FPN();
    head = SOLOHead();
    register_module("fpn", fpn);
    register_module("head", head);
}

SoloOut SOLOImpl::forward(const at::Tensor &input, bool eval)
{
    auto features = fpn->forward(input);
    return head->forward(features, eval);
}

SoloLoss SOLOImpl::loss(const SoloOut &pred, std::vector<Sample> &samples)
{
    auto &ins_preds = pred.ins_preds;
    auto &cate_preds = pred.cate_preds;

    // For each Pyramid Level
    // ToDO: analyze the statitics for hit indices
    vector<at::Tensor> ins_label_list, cate_label_list, ins_ind_label_list;
    int num_instances = 0; //variable for debug
    for (size_t i = 0; i < ins_preds.size(); i++)
    {
        int lower_bound = Cfg::scale_ranges[i][0];
        int upper_bound = Cfg::scale_ranges[i][1];
        int64_t height = ins_preds[i].size(2);
        int64_t width = ins_preds[i].size(3);
        int64_t upsampled_h = ins_preds[0].size(2) * 4;
        int64_t upsampled_w = ins_preds[0].size(3) * 4;
        int num_grid = Cfg::num_grids[i];

        vector<at::Tensor> ins_label_vec, cate_label_vec, ins_ind_label_vec;
        for (size_t batch_id = 0; batch_id < samples.size(); batch_id++)
        {
            auto gt_classes_raw = samples[batch_id].target.gt_classes;
            auto gt_bboxs_raw = samples[batch_id].target.gt_bboxs;
            auto gt_masks_raw = samples[batch_id].target.gt_masks;

            auto gt_areas = TorchUtil::bbox_area(gt_bboxs_raw);
            auto ins_label = torch::zeros({num_grid * num_grid, height, width}, torch::kUInt8).to(device);
            auto cate_label = torch::zeros({num_grid, num_grid}, torch::kInt64).to(device);
            auto ins_ind_label = torch::zeros({num_grid * num_grid}, torch::kBool).to(device);

            //!!! really important
            auto hit_indices = ((gt_areas >= lower_bound).bitwise_and(gt_areas <= upper_bound)).nonzero().narrow(1, 0, 1).flatten();
            if (false)
            {
                cout << "batch" << batch_id << " level" << i << " lower=" << lower_bound << " uppper_bound=" << upper_bound << endl;
                cout << "gt_areas: " << gt_areas << endl;
                cout << "hit_indices: " << hit_indices.size(0) << endl;
            }

            if (hit_indices.size(0) == 0)
            {
                ins_label_vec.push_back(ins_label);
                cate_label_vec.push_back(cate_label);
                ins_ind_label_vec.push_back(ins_ind_label.flatten());
                continue;
            }

            auto gt_bboxes = gt_bboxs_raw.index_select(0, hit_indices);
            auto gt_classes = gt_classes_raw.index_select(0, hit_indices);
            auto gt_masks = gt_masks_raw.index_select(0, hit_indices);

            // Center region should lies in the region of 0.2 * bbox_size
            auto half_ws = 0.5 * (gt_bboxes.narrow(1, 2, 1) - gt_bboxes.narrow(1, 0, 1)) * radius_scale;
            auto half_hs = 0.5 * (gt_bboxes.narrow(1, 3, 1) - gt_bboxes.narrow(1, 1, 1)) * radius_scale;

            // allocate the mask to corresponding grid
            for (int64_t hit_id = 0; hit_id < hit_indices.size(0); hit_id++)
            {
                auto seg_mask = gt_masks[hit_id];
                auto gt_cls = gt_classes[hit_id];
                int half_h = half_hs[hit_id].detach().cpu().item().toInt();
                int half_w = half_ws[hit_id].detach().cpu().item().toInt();
                int half_h_grid = half_h / (double)upsampled_h * num_grid;
                int half_w_grid = half_w / (double)upsampled_w * num_grid;

                if (seg_mask.sum().detach().cpu().item().toDouble() < 10)
                {
                    continue;
                }

                auto center = TorchUtil::center_of_mass(seg_mask);
                int center_h = center[0].detach().cpu().item().toInt();
                int center_w = center[1].detach().cpu().item().toInt();

                int coord_w = center_w / (double)upsampled_w * num_grid;
                int coord_h = center_h / (double)upsampled_h * num_grid;

                int top = coord_h - half_h_grid;
                top = top > 0 ? top : 0;
                top = top > (coord_h - 1) ? top : (coord_h - 1);
                int down = coord_h + half_h_grid;
                down = down < num_grid ? down : num_grid - 1;
                down = down < (coord_h + 1) ? down : (coord_h + 1);
                int left = coord_w - half_w_grid;
                left = left > 0 ? left : 0;
                left = left > (coord_w - 1) ? left : (coord_w - 1);
                int right = coord_w + half_w_grid;
                right = right < num_grid ? right : num_grid - 1;
                right = right < (coord_w + 1) ? right : (coord_w + 1);

                // category label
                cate_label.index_put_({Slice(top, down + 1), Slice(left, right + 1)}, gt_cls);

                // instance label
                double p_scale = p_scales[i];
                auto seg_mask_scale = seg_mask.unsqueeze(0).unsqueeze(0);
                seg_mask_scale = F::interpolate(seg_mask_scale, F::InterpolateFuncOptions().scale_factor(vector<double>({p_scale, p_scale})).mode(torch::kNearest).recompute_scale_factor(true));
                seg_mask_scale = seg_mask_scale.squeeze(0).squeeze(0);
                for (int row = top; row < down + 1; row++)
                {
                    for (int col = left; col < right + 1; col++)
                    {
                        int label = row * num_grid + col;
                        ins_label[label] = seg_mask_scale;
                        ins_ind_label[label] = true;
                        num_instances++;
                        if (false)
                        {
                            cv::Mat ins_label_img = ImgUtil::TensorToCvMat(ins_label[label].to(at::kFloat));
                            ins_label_img = ins_label_img * 255.0;
                            ins_label_img.convertTo(ins_label_img, CV_8UC1);
                            cv::imshow(to_string(row) + "_" + to_string(col), ins_label_img);
                            cv::waitKey(-1);
                        }
                    }
                }

                if (false)
                {
                    cout << "bbox:" << gt_bboxes[hit_id] << " mask_scaled:" << seg_mask_scale.sizes() << " ins_pred:" << ins_preds[i].sizes() << endl;
                    cout << "center=" << center_w << "," << center_h << ", coord=" << coord_w << "," << coord_h << endl;
                    cout << "top " << top << " down " << down << " left " << left << " right " << right << endl;
                    cv::Mat mask = ImgUtil::TensorToCvMat(seg_mask);
                    mask = mask * 255.0;
                    mask.convertTo(mask, CV_8UC1);
                    cv::Mat color_img;
                    cv::merge(vector<cv::Mat>({mask, mask, mask}), color_img);
                    cv::circle(color_img, cv::Point(center_w, center_h), 3, cv::Scalar(0, 0, 255), -1);
                    cv::rectangle(color_img, cv::Rect(center_w - half_w, center_h - half_h, half_w * 2, half_h * 2), cv::Scalar(255, 0, 0));
                    cv::imshow(to_string(batch_id) + "_level" + to_string(i), color_img);
                    cv::Mat mask_scale = ImgUtil::TensorToCvMat(seg_mask_scale);
                    mask_scale = mask_scale * 255.0;
                    mask_scale.convertTo(mask_scale, CV_8UC1);
                    cv::imshow(to_string(batch_id) + "_level" + to_string(i) + "_scale", mask_scale);
                    cv::waitKey(-1);
                }
            }

            ins_label_vec.push_back(ins_label);
            cate_label_vec.push_back(cate_label);
            ins_ind_label_vec.push_back(ins_ind_label.flatten());
        }

        ins_label_list.push_back(torch::cat(ins_label_vec, 0));
        cate_label_list.push_back(torch::cat(cate_label_vec, 0));
        ins_ind_label_list.push_back(torch::cat(ins_ind_label_vec));
    }

    auto ins_ind_label_batch = torch::cat(ins_ind_label_list);

    // Instance Loss
    vector<at::Tensor> loss_ins_vec; //ToDo: change to Batch Mode
    for (size_t i = 0; i < ins_preds.size(); i++)
    {
        auto mask_idxs = ins_ind_label_list[i].nonzero().flatten();
        if (mask_idxs.size(0) == 0)
            continue;
        auto gt_instance = ins_label_list[i].index_select(0, mask_idxs);
        auto pred_instance = ins_preds[i].view({-1, ins_preds[i].size(2), ins_preds[i].size(3)});
        pred_instance = pred_instance.index_select(0, mask_idxs);
        pred_instance = torch::sigmoid(pred_instance); //!!! very important
        auto loss_ins_i = dice_loss(pred_instance, gt_instance);
        loss_ins_vec.push_back(loss_ins_i);
    }
    auto loss_ins = torch::cat(loss_ins_vec).mean();

    // category loss is arranged in level-batch sense, for each level, there are #batch_sizes images
    vector<at::Tensor> cate_pred_vec, cate_label_vec;
    for (size_t i = 0; i < cate_preds.size(); i++)
    {
        // cout <<
        cate_pred_vec.push_back(cate_preds[i].permute({0, 2, 3, 1}).reshape({-1, Cfg::num_classes - 1}));
        cate_label_vec.push_back(cate_label_list[i].flatten());
    }
    auto cate_preds_batch = torch::cat(cate_pred_vec, 0);
    auto cate_label_batch = torch::cat(cate_label_vec);
    num_instances = ins_ind_label_batch.sum().detach().cpu().item().toInt();
    auto loss_cate = sigmoid_focal_loss(cate_preds_batch, cate_label_batch, num_instances);

    return {loss_ins, loss_cate};
}
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <ImgUtil.h>

#include "config.h"
#include "solo.h"

using namespace std;
using namespace torch::indexing;
namespace fs = boost::filesystem;

void predict()
{
    SoloDataset train_set("train.txt");
    SoloDataset val_set("train.txt");
    cout << "====================Train Set Size=" << train_set.size().value() << ", val set size=" << val_set.size().value() << endl;
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        train_set,
        torch::data::DataLoaderOptions().batch_size(1).workers(0));
    auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        val_set,
        torch::data::DataLoaderOptions().batch_size(1).workers(0));

    SOLO solo;
    solo->to(device);
    torch::load(solo, "./output/solo_final.pt");
    solo->eval();
    torch::NoGradGuard();
    int step = 0;
    for (int epoch = 0; epoch < Cfg::epoch; epoch++)
    {
        for (auto &input : *train_loader)
        {
            std::vector<torch::Tensor> images;
            std::vector<torch::Tensor> gt_bboxs;
            std::vector<torch::Tensor> gt_masks;
            std::vector<torch::Tensor> gt_classes;
            for (size_t i = 0; i < input.size(); i++)
            {
                input[i].data.image = input[i].data.image.to(device);
                input[i].target.gt_classes = input[i].target.gt_classes.to(device);
                input[i].target.gt_bboxs = input[i].target.gt_bboxs.to(device);
                input[i].target.gt_masks = input[i].target.gt_masks.to(device);

                images.push_back(input[i].data.image);
                gt_bboxs.push_back(input[i].target.gt_bboxs);
                gt_masks.push_back(input[i].target.gt_masks); // #views x H x W
                gt_classes.push_back(input[i].target.gt_classes);
            }
            //assert(gt_views.size() == Cfg::batch_size);

            auto img_batch = torch::stack(images); // N x 2 x H x W for input_views = "FS"
            auto gt_bbox_batch = torch::cat(gt_bboxs);
            auto gt_mask_batch = torch::cat(gt_masks);
            auto gt_class_batch = torch::cat(gt_classes);
            // auto contour_batch = torch::cat(contours);

            if (false)
            {
                cout << "inputs: " << img_batch.sizes() << endl;
                cout << "gt_bbox_batch: " << gt_bbox_batch.sizes() << endl;
                cout << "gt_mask_batch: " << gt_mask_batch.sizes() << endl;
                cout << "gt_class_batch: " << gt_class_batch.sizes() << endl;
            }

            SoloOut pred = solo->forward(img_batch, true); // 2 x 5 x H x W
            std::vector<at::Tensor> &cate_pred_vec = pred.cate_preds;
            std::vector<at::Tensor> &ins_pred_vec = pred.ins_preds;
            for (size_t l = 0; l < cate_pred_vec.size(); l++)
            {
                cate_pred_vec[l] = cate_pred_vec[l].reshape({-1, Cfg::num_classes - 1});
                ins_pred_vec[l] = ins_pred_vec[l].squeeze(0);
                cout << l << " " << cate_pred_vec[l].sizes() << " ins: " << ins_pred_vec[l].sizes() << endl;
            }
            at::Tensor cate_preds = torch::cat(cate_pred_vec, 0);
            at::Tensor ins_preds = torch::cat(ins_pred_vec, 0);

            // category scores and labels
            auto ind = (cate_preds > Cfg::score_thr).nonzero();
            auto inds = ind.narrow(1, 0, 1).flatten();
            auto cate_labels = ind.narrow(1, 1, 1).flatten();
            auto cate_scores = cate_preds.index({inds, cate_labels});

            if (false)
            {
                cout << "cate_preds: " << cate_preds.sizes() << endl;
                cout << "ins_preds: " << ins_preds.sizes() << endl;
                cout << "ind: " << ind.sizes() << endl;
                cout << "scores: " << cate_scores.sizes() << endl;
                cout << "cate_labels: " << cate_labels.sizes() << endl;
                cout << cate_labels << endl;
            }

            if (cate_scores.size(0) == 0)
            {
                cout << "No instances detected!" << endl;
                return;
            }

            // strides
            auto size_trans = torch::tensor(Cfg::num_grids).pow(2).cumsum(0);
            int num_grids_l1 = size_trans[0].detach().cpu().item().toInt();
            int sum_grids = size_trans[-1].detach().cpu().item().toInt(); // sum of grids number across all pyramid level
            auto strides = torch::ones({sum_grids}).to(device);
            strides.index_put_({Slice(0, num_grids_l1)}, strides.index({Slice(0, num_grids_l1)}) * Cfg::strides[0]);
            for (size_t l = 1; l < Cfg::strides.size(); l++)
            {
                int start = size_trans[l - 1].detach().cpu().item().toInt();
                int end = size_trans[l].detach().cpu().item().toInt();
                strides.index_put_({Slice(start, end)}, strides.index({Slice(start, end)}) * Cfg::strides[l]);
            }
            strides = strides.index_select(0, inds);
            if (false)
            {
                cout << "size_trans: " << size_trans << endl;
                cout << "sum_grids: " << sum_grids << endl;
                cout << "strides: " << strides << endl;
            }

            // masks
            auto seg_preds = ins_preds.index_select(0, inds);
            auto seg_masks = seg_preds > Cfg::mask_thr;
            auto sum_masks = seg_masks.sum({1, 2}).toType(at::kFloat);
            // filter small masks
            auto keep = (sum_masks > strides);
            if (keep.sum().detach().cpu().item().toInt() == 0)
            {
                cout << "No mask detected!" << endl;
                return;
            }
            auto keep_inds = keep.nonzero().flatten();
            seg_masks = seg_masks.index_select(0, keep_inds);
            seg_preds = seg_preds.index_select(0, keep_inds);
            sum_masks = sum_masks.index_select(0, keep_inds);
            cate_scores = cate_scores.index_select(0, keep_inds);
            cate_labels = cate_labels.index_select(0, keep_inds);

            if (true)
            {
                string save_dir = "./result/";
                if (!fs::exists(save_dir))
                {
                    fs::create_directories(save_dir);
                }
                cout << "sum_masks: " << sum_masks.sizes() << endl;
                cout << "keep_inds: " << keep_inds.sizes() << endl;
                cout << "seg_masks: " << seg_masks.sizes() << endl;
                cout << "seg_preds: " << seg_preds.sizes() << endl;
                cout << "cate_scores: " << cate_scores.sizes() << endl;
                cout << "cate_labels: " << cate_labels.sizes() << endl;
                for (int i = 0; i < seg_masks.size(0); i++)
                {
                    int cls = cate_labels[i].detach().cpu().item().toInt();
                    auto seg_mask = seg_masks[i].toType(at::kFloat).detach().cpu();
                    auto seg_img = ImgUtil::TensorToMaskMat(seg_mask);
                    cv::imwrite(save_dir + to_string(i) + "_" + Cfg::class_names[cls + 1] + ".jpg", seg_img);
                }
            }

            // mask scoring
            auto seg_scores = (seg_preds * seg_masks.toType(at::kFloat)).sum({1, 2}) / sum_masks;
            cate_scores *= seg_scores;
            // sort and keep nms_pre
            auto sort_inds = cate_scores.argsort(0, true);
            if (cate_scores.size(0) > Cfg::nms_pre)
            {
                sort_inds = sort_inds.index({Slice(0, Cfg::nms_pre)});
            }
            seg_masks = seg_masks.index_select(0, sort_inds);
            seg_preds = seg_preds.index_select(0, sort_inds);
            sum_masks = sum_masks.index_select(0, sort_inds);
            cate_scores = cate_scores.index_select(0, sort_inds);
            cate_labels = cate_scores.index_select(0, sort_inds);

            if (true)
            {
                cout << "sorted_scores\n";
                cout << cate_scores << endl;
            }
            break;
        }
        break;
    }
}

int main()
{
    predict();
    return 0;
}
#include <iostream>
#include <TimeUtil.h>
#include <RandUtil.h>
#include "solo.h"
#include "config.h"

using namespace std;

void train()
{
    SoloDataset train_set("train.txt");
    SoloDataset val_set("val.txt");
    int max_iter_val = val_set.size().value();
    cout << "====================Train Set Size=" << train_set.size().value() << ", val set size=" << val_set.size().value() << endl;

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        train_set,
        torch::data::DataLoaderOptions().batch_size(Cfg::batch_size).workers(12));
    auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        val_set,
        torch::data::DataLoaderOptions().batch_size(2).workers(12));

    SOLO solo;
    solo->to(device);

    torch::optim::Adam optim(solo->parameters(),
                             torch::optim::AdamOptions(Cfg::lr).weight_decay(Cfg::weight_decay));
    TimeUtil::Timer timer;
    double data_time, forward_time, backward_time;
    int step = 0;
    for (int epoch = 0; epoch < Cfg::epoch; epoch++)
    {
        solo->train();
        timer.reset();
        for (auto &input : *train_loader)
        {
            optim.zero_grad();
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
            data_time = timer.elapsed();

            auto img_batch = torch::stack(images); // N x 2 x H x W for input_views = "FS"
            auto gt_bbox_batch = torch::cat(gt_bboxs);
            auto gt_mask_batch = torch::cat(gt_masks);
            auto gt_class_batch = torch::cat(gt_classes);

            if (false)
            {
                cout << "inputs: " << img_batch.sizes() << endl;
                cout << "gt_bbox_batch: " << gt_bbox_batch.sizes() << endl;
                cout << "gt_mask_batch: " << gt_mask_batch.sizes() << endl;
                cout << "gt_class_batch: " << gt_class_batch.sizes() << endl;
            }

            //-------------------forward------------------
            timer.reset();
            SoloOut pred = solo->forward(img_batch);
            forward_time = timer.elapsed();

            //-------------------backward------------------
            timer.reset();
            SoloLoss solo_loss = solo->loss(pred, input);
            auto loss = Cfg::lambda_ins * solo_loss.ins_loss + Cfg::lambda_cat * solo_loss.cate_loss;
            loss.backward();
            optim.step();
            backward_time = timer.elapsed();

            float loss_ = loss.detach().cpu().item().toFloat();
            float ins_loss_ = solo_loss.ins_loss.detach().cpu().item().toFloat();
            float cate_loss_ = solo_loss.cate_loss.detach().cpu().item().toFloat();
            printf("[%2d/%2d][%3d] loss %.4f |mask %.4f |class %.4f  |Time data:%.4fs, forward:%.4f, backward:%.4f\n",
                   epoch, Cfg::epoch, step, loss_, ins_loss_, cate_loss_, data_time, forward_time, backward_time);
            step++;
        }

        int iter = 0;
        if (epoch % Cfg::val_epoch == 0)
        {
            int vis_iter = RandUtil::randint(1, max_iter_val - 1);
            solo->eval();
            torch::NoGradGuard();
            for (auto &input : *val_loader)
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
                auto img_batch = torch::stack(images); // N x 3 x H x W for input_views = "FS"
                auto gt_bbox_batch = torch::cat(gt_bboxs);
                auto gt_mask_batch = torch::cat(gt_masks);
                auto gt_class_batch = torch::cat(gt_classes);

                SoloOut head_out = solo->forward(img_batch);
                SoloLoss solo_loss = solo->loss(head_out, input);
                auto loss = Cfg::lambda_ins * solo_loss.ins_loss + Cfg::lambda_cat * solo_loss.cate_loss;

                float loss_ = loss.detach().cpu().item().toFloat();
                float ins_loss_ = solo_loss.ins_loss.detach().cpu().item().toFloat();
                float cate_loss_ = solo_loss.cate_loss.detach().cpu().item().toFloat();
                printf("Eval [%2d/%2d][%3d] loss %.4f |mask %.4f |class %.4f\n",
                       epoch, Cfg::epoch, iter, loss_, ins_loss_, cate_loss_);

                // visualization
                if (iter == 0 || iter == vis_iter)
                {
                    SoloOut vis_sample;
                    for (size_t l = 0; l < head_out.ins_preds.size(); l++)
                    {
                        vis_sample.ins_preds.push_back(head_out.ins_preds[l][0].unsqueeze(0));
                        vis_sample.cate_preds.push_back(head_out.cate_preds[l][0].unsqueeze(0));
                    }

                    SoloPred pred;
                    if (solo->post_process(vis_sample, pred))
                    {
                        string save_dir = Cfg::output_dir + to_string(epoch) + "/" + to_string(iter) + "/";
                        solo->visualize_input(input[0].data.image, save_dir);
                        solo->visualize_pred(pred, save_dir + "predict/");
                    }
                }
                if (!fs::exists(Cfg::output_dir))
                {
                    fs::create_directories(Cfg::output_dir);
                }
                torch::save(solo, Cfg::output_dir + "solo_" + to_string(epoch) + ".pt");
                iter++;
            }

            torch::save(solo, Cfg::output_dir + "solo_final.pt");
        }
    }
}

int main()
{
    train();
    return 0;
}
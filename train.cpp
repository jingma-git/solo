#include <iostream>
#include <TimeUtil.h>
#include "solo.h"
#include "config.h"

using namespace std;

void train()
{
    SoloDataset train_set("train.txt");
    SoloDataset val_set("train.txt");
    cout << "====================Train Set Size=" << train_set.size().value() << ", val set size=" << val_set.size().value() << endl;
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        train_set,
        torch::data::DataLoaderOptions().batch_size(Cfg::batch_size).workers(12));
    auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        val_set,
        torch::data::DataLoaderOptions().batch_size(Cfg::batch_size).workers(12));

    SOLO solo;
    solo->to(device);

    torch::optim::Adam optim(solo->parameters(),
                             torch::optim::AdamOptions(Cfg::lr).weight_decay(Cfg::weight_decay));
    TimeUtil::Timer timer;
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

            SoloOut pred = solo->forward(img_batch); // 2 x 5 x H x W
            SoloLoss solo_loss = solo->loss(pred, input);
            auto loss = Cfg::lambda_ins * solo_loss.ins_loss + Cfg::lambda_cat * solo_loss.cate_loss;
            loss.backward();
            optim.step();

            float loss_ = loss.detach().cpu().item().toFloat();
            float ins_loss_ = solo_loss.ins_loss.detach().cpu().item().toFloat();
            float cate_loss_ = solo_loss.cate_loss.detach().cpu().item().toFloat();
            printf("[%2d/%2d][%3d] loss %.4f |mask %.4f |class %.4f  |Time %.4fs\n",
                   epoch, Cfg::epoch, step, loss_, ins_loss_, cate_loss_, timer.elapsed());
            step++;
        }

        torch::save(solo, Cfg::output_dir + "solo_final.pt");
    }
}

int main()
{
    train();
    return 0;
}
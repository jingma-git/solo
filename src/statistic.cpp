#include "statistic.h"
#include "dataset.h"
#include "TorchUtil.h"
#include "config.h"

#include <iostream>
using namespace std;

namespace StatUtil
{
    void analyze_dataset()
    {
        SoloDataset train_set("train.txt");

        cout << "====================Train Set Size=" << train_set.size().value() << endl;

        auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            train_set,
            torch::data::DataLoaderOptions().batch_size(2).workers(12));
        std::vector<int> num_instances(Cfg::num_classes, 0);
        for (auto &samples : *train_loader)
        {
            for (size_t batch_id = 0; batch_id < samples.size(); batch_id++)
            {
                auto &gt_classes_raw = samples[batch_id].target.gt_classes;
                for (int cls_id = 0; cls_id < Cfg::num_classes; cls_id++)
                {
                    num_instances[cls_id] += (gt_classes_raw == cls_id).sum().detach().cpu().item().toInt();
                }
            }
        }

        cout << "-------instances amount for each category--------------" << endl;
        for (int cls_id = 0; cls_id < Cfg::num_classes; cls_id++)
        {
            cout << Cfg::class_names[cls_id] << ": " << num_instances[cls_id] << endl;
        }

        std::vector<double> bbox_areas(Cfg::num_classes, 0.0);
        for (auto &samples : *train_loader)
        {
            for (size_t batch_id = 0; batch_id < samples.size(); batch_id++)
            {
                auto &gt_classes_raw = samples[batch_id].target.gt_classes;
                auto &gt_bboxs_raw = samples[batch_id].target.gt_bboxs;

                auto gt_areas = TorchUtil::bbox_area(gt_bboxs_raw);
                for (int i = 0; i < gt_classes_raw.size(0); i++)
                {
                    int cls_id = gt_classes_raw[i].detach().cpu().item().toInt();
                    cout << Cfg::class_names[cls_id] << ": " << gt_areas[i].detach().cpu().item().toDouble() << endl;
                    double num_ins = num_instances[cls_id];
                    if (num_ins > 0)
                        bbox_areas[cls_id] += gt_areas[i].detach().cpu().item().toDouble() / num_ins;
                }
            }
        }

        cout << "-------instances amount for each category--------------" << endl;
        for (int cls_id = 0; cls_id < Cfg::num_classes; cls_id++)
        {
            cout << Cfg::class_names[cls_id] << ": " << bbox_areas[cls_id] << endl;
        }
    }
} // namespace StatUtil

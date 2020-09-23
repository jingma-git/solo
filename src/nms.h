#pragma once
#include <torch/torch.h>
#include <string>

namespace F = torch::nn::functional;
using namespace torch::indexing;
inline at::Tensor point_nms(const at::Tensor &heat, int kernel = 2)
{
     auto hmax = F::max_pool2d(heat, F::MaxPool2dFuncOptions(2).stride(1).padding(1));
     auto keep = (hmax.index({Slice(), Slice(), Slice(None, -1), Slice(None, -1)}) == heat).toType(at::kFloat);
     return heat * keep;
}

// Matrix NMS for multi-class masks.

//     Args:
//         seg_masks (Tensor): shape (n, h, w)
//         cate_labels (Tensor): shape (n), mask labels in descending order
//         cate_scores (Tensor): shape (n), mask scores in descending order
//         sum_masks (Tensor): The sum of seg_masks
//         kernel (str):  'linear' or 'gauss'
//         sigma (float): std in gaussian method

//     Returns:
//         Tensor: cate_scores_update, tensors of shape (n)

inline at::Tensor matrix_nms(const at::Tensor &seg_masks,
                             const at::Tensor &cate_labels,
                             const at::Tensor &cate_scores,
                             const at::Tensor &sum_masks,
                             std::string kernel = "gaussian",
                             double sigma = 2.0)
{
     using namespace std;
     using namespace torch::indexing;

     int n_samples = cate_labels.size(0);
     if (n_samples == 0)
     {
          return at::Tensor();
     }
     auto seg_mask = seg_masks.reshape({n_samples, -1}).toType(at::kFloat);
     auto inter_matrix = torch::mm(seg_mask, seg_mask.transpose(1, 0));
     auto sum_masks_x = sum_masks.expand({n_samples, n_samples});
     auto iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(1);
     auto cate_labels_x = cate_labels.expand({n_samples, n_samples});
     //1 means the sample_i and sample_j belongs to the same category, suppression are performed on entries whose value==1
     auto label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).toType(at::kFloat).triu(1);

     at::Tensor compensate_iou, _;
     //IoU compensation, record the "max_iou" for each sample_j
     std::tie(compensate_iou, _) = (iou_matrix * label_matrix).max(0);
     compensate_iou = compensate_iou.expand({n_samples, n_samples}).transpose(1, 0);

     auto decay_iou = iou_matrix * label_matrix;
     at::Tensor decay_coeff;
     if (kernel == "gaussian")
     {
          auto decay_matrix = torch::exp(-1 * sigma * (decay_iou * decay_iou));
          auto compensate_matrix = torch::exp(-1 * sigma * (compensate_iou * compensate_iou));
          std::tie(decay_coeff, _) = (decay_matrix / compensate_matrix).min(0);
     }
     else
     {
          auto decay_matrix = (1 - decay_iou) / (1 - compensate_iou);
          std::tie(decay_coeff, _) = decay_matrix.min(0);
     }
     at::Tensor cate_scores_update = cate_scores * decay_coeff;

     if (false)
     {
          cout << "mask matrix:\n"
               << sum_masks_x.index({Slice(0, 10), Slice(0, 10)}) << endl;
          cout << "iou_matrix:\n"
               << iou_matrix.index({Slice(0, 10), Slice(0, 10)}) << endl;
          cout << "cate_labels_x:\n"
               << cate_labels_x.index({Slice(0, 10), Slice(0, 10)}) << endl;
          cout << cate_labels_x.transpose(1, 0).index({Slice(0, 10), Slice(0, 10)}) << endl;
          cout << "label_matrix:\n"
               << label_matrix.index({Slice(0, 10), Slice(0, 10)}) << endl;
          cout << "compensate_iou:\n"
               << compensate_iou.index({Slice(0, 10), Slice(0, 10)}) << endl;
          cout << "cate_scores:\n";
          cout << cate_scores << endl;
          cout << "decay_coeff:\n";
          cout << decay_coeff << endl;
          cout << "cate_scores_update:\n";
          cout << cate_scores_update << endl;
     }

     return cate_scores_update;
}

#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "fpn.h"
#include "loss.h"
#include "sigmoid_focal_loss.h"
#include "solo_head.h"
#include "conv_module.h"
#include "config.h"
#include "ImgUtil.h"
#include "dataset.h"
#include "TorchUtil.h"

#ifdef WITH_CUDA
#include "sigmoid_focal_loss.h"
#endif

using namespace std;

void testCenterOfMass()
{
    cv::Mat mask = cv::Mat::zeros(512, 512, CV_8UC1);
    cv::rectangle(mask, cv::Rect(200, 300, 200, 200), cv::Scalar(255), -1);
    mask.convertTo(mask, CV_32F);
    mask = mask / 255.0;
    at::Tensor mask_tensor = ImgUtil::CvImageToTensor(mask);
    cout << "center: " << TorchUtil::center_of_mass(mask_tensor) << endl;
}

void testNet()
{
    cv::Mat mask = cv::Mat::zeros(512, 512, CV_8UC1);
    cv::rectangle(mask, cv::Rect(200, 300, 200, 200), cv::Scalar(255), -1);
    mask.convertTo(mask, CV_32F);
    mask = mask / 255.0;
    at::Tensor mask_tensor = ImgUtil::CvImageToTensor(mask);

    auto x = torch::randn({2, 3, 512, 512});
    auto gt_bbox = torch::tensor({{200, 300, 200, 200}, {200, 300, 200, 200}});
    auto gt_label = torch::tensor({2, 2});
    auto gt_mask = torch::stack({mask_tensor, mask_tensor});
    cout << "gt_mask: " << gt_mask.sizes() << endl;
    FPN fpn;
    auto features = fpn->forward(x);

    SOLOHead head;
    SoloOut solo_out = head->forward(features);
}

void statistics()
{
}

void testIndex()
{
    auto x = torch::arange(10);
    using namespace torch::indexing;
    cout << x << endl;
    x.index_put_({Slice(1, 3)}, 200);

    cout << x << endl;
}

void testInterpolate()
{
    auto x = torch::randn({8, 8});
    x = x.unsqueeze(0).unsqueeze(0);
    auto y = F::interpolate(x, F::InterpolateFuncOptions().scale_factor(vector<double>({0.5, 0.5})).mode(torch::kNearest));
    cout << "x\n"
         << x << endl;
    cout << "y\n"
         << y << endl;
}

void testDiceLoss()
{
    int N = 2;
    int M = 100;
    auto gt = torch::zeros({N, M});
    auto pred = torch::zeros({N, M});
    gt[0][0] = 0.1;
    pred[0][0] = 1;
    auto loss = dice_loss(pred, gt);
    cout << loss << endl;
}

void testFocalLoss()
{
    int N = 1;
    int M = 16;
    auto gt = torch::zeros({N, M});
    auto pred = torch::zeros({N, M});
    // gt[0][0] = 1;
    gt[0][2] = 1;
    pred[0][0] = 1;

    auto py_loss = py_sigmoid_focal_loss(pred, gt);
    cout << "py_sigmoid_focal_loss: " << py_loss << endl;

#ifdef WITH_CUDA
    auto target = torch::tensor({3}, at::kLong);
    auto cuda_loss = SigmoidFocalLossFunction::apply(pred.to(device), target.to(device));
    cout << "cuda_sigmoid_focal_loss: " << cuda_loss << endl;
#endif
}

void testSum()
{
    auto x = torch::zeros({2, 3, 3});
    x[0][0][0] = 1;
    x[0][0][1] = 2;
    cout << "x" << endl;
    cout << x << endl;
    cout << "x sum" << endl;
    cout << x.sum({1, 2}) << endl;
}

int main()
{

    // testNet();
    // testCenterOfMass();
    // testIndex();
    // testInterpolate();
    // testDiceLoss();
    // testFocalLoss();
    testSum();
    return 0;
}
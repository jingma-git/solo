#pragma once
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace ImgUtil
{
    cv::Mat normalize_img(const cv::Mat &img);
    cv::Mat unnormalize_img(const cv::Mat &img, float maxval = 255.0);

    // 	input:
    // 		image:   n x H x W x C : images with value range [-1.0, 1.0] in each channel
    // 	output:
    // 		mask:    n x H x W x 1 : boolean mask (depth channel value < 0.9)
    cv::Mat extract_boolean_mask(const cv::Mat &img);

    // 	input:
    // 		image:   n x H x W x C : images with value range [-1.0, 1.0] in each channel
    // 	output:
    // 		depth:    n x H x W x 1 : boolean mask (depth channel value < 0.9)
    void extract_depth_normal(const cv::Mat &img, cv::Mat &normal, cv::Mat &depth);

    at::Tensor CvImageToTensor(const cv::Mat &image);

    cv::Mat TensorToCvMat(const torch::Tensor &tensor);

    // input:
    //      tensor: 1xHxW or HxW torch float tensor with range [0, 1.0]
    cv::Mat TensorToMaskMat(const torch::Tensor &tensor);
} // namespace ImgUtil

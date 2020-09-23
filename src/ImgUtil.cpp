#include "ImgUtil.h"
#include <CvUtil.h>

namespace ImgUtil
{
    cv::Mat normalize_img(const cv::Mat &img)
    {
        cv::Mat ret;
        img.convertTo(ret, CV_32FC3);

        // normalize to [0.0, 1.0]
        if (CvUtil::dataType(img.type()) == "8U")
        {
            ret = ret / 255.0;
        }
        else if (CvUtil::dataType(img.type()) == "16U")
        {
            ret = ret / 65535.0;
        }

        return ret;
    }

    cv::Mat unnormalize_img(const cv::Mat &img, float maxval)
    {
        cv::Mat ret = img.clone();
        ret = ret * maxval;
        return ret;
    }

    cv::Mat extract_boolean_mask(const cv::Mat &img)
    {
        cv::Mat bgra[4];
        cv::split(img, bgra);
        return (bgra[3] < 0.9) / 255;
    }

    void extract_depth_normal(const cv::Mat &img, cv::Mat &normal, cv::Mat &depth)
    {
        cv::Mat bgra[4];
        cv::split(img, bgra);
        std::vector<cv::Mat> bgr({bgra[0], bgra[1], bgra[2]});
        cv::merge(bgr, normal);
        depth = bgra[3];
    }

    at::Tensor CvImageToTensor(const cv::Mat &image)
    {
        // Idea taken from https://github.com/pytorch/pytorch/issues/12506
        // we have to split the interleaved channels
        cv::Mat channelsConcatenatedFloat;
        if (image.channels() == 4)
        {
            cv::Mat bgra[4];
            cv::split(image, bgra);
            cv::Mat channelsConcatenated;
            cv::vconcat(bgra[2], bgra[1], channelsConcatenated);
            cv::vconcat(channelsConcatenated, bgra[0], channelsConcatenated);
            cv::vconcat(channelsConcatenated, bgra[3], channelsConcatenated);

            channelsConcatenated.convertTo(channelsConcatenatedFloat, CV_32FC4);
            assert(channelsConcatenatedFloat.isContinuous());
        }
        else if (image.channels() == 3)
        {
            cv::Mat bgr[3];
            cv::split(image, bgr);
            cv::Mat channelsConcatenated;
            cv::vconcat(bgr[2], bgr[1], channelsConcatenated);
            cv::vconcat(channelsConcatenated, bgr[0], channelsConcatenated);

            channelsConcatenated.convertTo(channelsConcatenatedFloat, CV_32FC3);
            assert(channelsConcatenatedFloat.isContinuous());
        }
        else if (image.channels() == 1)
        {
            image.convertTo(channelsConcatenatedFloat, CV_32FC1);
        }
        else
        {
            throw std::invalid_argument("CvImageToTensor: Unsupported image format");
        }

        at::TensorOptions options(at::kFloat);
        at::Tensor tensor_image =
            torch::from_blob(channelsConcatenatedFloat.data, {image.channels(), image.rows, image.cols},
                             options.requires_grad(false))
                .clone(); // clone is required to copy data from temporary object
        return tensor_image.squeeze();
    }

    cv::Mat TensorToCvMat(const torch::Tensor &tensor)
    {
        int channels;
        int H;
        int W;
        if (tensor.sizes().size() == 2)
        {
            channels = 1;
            H = static_cast<int>(tensor.size(0));
            W = static_cast<int>(tensor.size(1));
        }
        else
        {
            channels = static_cast<int>(tensor.size(0));
            H = static_cast<int>(tensor.size(1));
            W = static_cast<int>(tensor.size(2));
        }

        cv::Mat ret;
        if (channels == 1)
        {
            auto tmp = tensor;
            cv::Mat img(H, W, CV_32FC1, tmp.data_ptr<float>());
            ret = img.clone();
        }
        else if (channels == 3)
        {
            std::vector<torch::Tensor> tmp = tensor.split(1, 0);
            cv::Mat b(H, W, CV_32FC1, tmp[0].data_ptr<float>());
            cv::Mat g(H, W, CV_32FC1, tmp[1].data_ptr<float>());
            cv::Mat r(H, W, CV_32FC1, tmp[2].data_ptr<float>());

            std::vector<cv::Mat> bgr({r, g, b});
            cv::merge(bgr, ret);
        }
        else if (channels == 4)
        {
            std::vector<torch::Tensor> tmp = tensor.split(1, 0);
            cv::Mat b(H, W, CV_32FC1, tmp[0].data_ptr<float>());
            cv::Mat g(H, W, CV_32FC1, tmp[1].data_ptr<float>());
            cv::Mat r(H, W, CV_32FC1, tmp[2].data_ptr<float>());
            cv::Mat a(H, W, CV_32FC1, tmp[3].data_ptr<float>());

            std::vector<cv::Mat> bgra({r, g, b, a});
            cv::merge(bgra, ret);
        }

        return ret;
    }

    cv::Mat TensorToMaskMat(const torch::Tensor &tensor)
    {
        cv::Mat mat = TensorToCvMat(tensor);
        mat = mat * 255.0;
        mat.convertTo(mat, CV_8UC1);
        return mat;
    }

} // namespace ImgUtil

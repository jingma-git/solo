#include "loss.h"

#ifdef WITH_CUDA
#include "sigmoid_focal_loss.h"
#endif

#include <iostream>
using namespace std;

at::Tensor reduced_dice_loss(const at::Tensor &pred, const at::Tensor &target, std::string reduction)
{
    if (reduction != "mean")
    {
        cerr << "Dice Loss without mean reduction have not been implemented!" << endl;
        exit(-1);
    }

    auto loss = dice_loss(pred, target);
    loss = loss.mean();

    return loss;
}

at::Tensor sigmoid_focal_loss(const at::Tensor &pred,
                              const at::Tensor &target,
                              int avg_factor,
                              std::string reduction,
                              double gamma,
                              double alpha)
{
#ifdef WITH_CUDA
    auto loss = SigmoidFocalLossFunction::apply(pred, target);
    if (reduction == "mean")
    {
        loss = loss.sum() / avg_factor;
    }
    else if (reduction == "sum")
    {
        loss = loss.sum();
    }
    return loss;
#else
    cerr << "Focal Loss with cpu support has not been implemented!" << endl;
    exit(-1);
#endif
}
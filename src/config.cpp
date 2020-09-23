#include "config.h"

using namespace std;

std::vector<std::string> Cfg::class_names = {
    "BG",
    "FG", //whole body
    "HEAD",
    "REAR",
    "LEAR",
    "NOSE",
    "TORSO",
    "RARM",
    "LARM",
    "RHAND", // rarely happen
    "LHAND",
    "RLEG",
    "LLEG",
    "RFOOT",
    "LFOOT",
    "TAIL",

    "LWING", // for bird
    "RWING"};

//************************Model**************************
//
bool Cfg::pretrain = false;
// ResNet
int Cfg::resnet_arch = 50;
int Cfg::in_channels = 1;
// SOLOHead
int Cfg::num_classes = Cfg::class_names.size();
std::vector<int> Cfg::num_grids = {40, 36, 24, 16, 12};
std::vector<std::vector<int>> Cfg::scale_ranges =
    // {{1, 96}, {48, 192}, {96, 384}, {192, 768}, {384, 2048}};
    {{8, 32}, {16, 64}, {32, 128}, {64, 256}, {128, 512}};
int Cfg::stacked_convs = 7;
int Cfg::num_groups = 32;
std::vector<int> Cfg::strides = {8, 8, 16, 32, 32};
// Loss
std::string Cfg::loss_cat = "focal_loss";
std::string Cfg::loss_ins = "dice_loss";
double Cfg::lambda_cat = 1.0;
double Cfg::lambda_ins = 3.0;
// Multi-view
int Cfg::img_size = 512;
int Cfg::input_size = 512; // the input image size for the network
double Cfg::img_scale = Cfg::input_size / (double)Cfg::img_size;
std::vector<std::string> input_view_names = {"F", "S"};
std::vector<std::string> Cfg::target_view_names = {
    "Front",   // T_0 Target_0  camera_position=(0, 0, 1.5)
    "Back",    // T_1  (0, 0, -1.5)
    "Left",    //(-1.5, 0, 0)
    "Right",   //(1.5, 0, 0)
    "Top",     //(0.0, 1.5, 0)
    "Bottom"}; //(0.0, -1.5, 0)
std::string Cfg::input_views = "F";
int Cfg::num_target_views = 1;
int Cfg::num_views = 1;
string Cfg::style_ids = "12345";

//************************Data**************************
string Cfg::output_dir = "./output/";
string Cfg::data_dir = "data/EasyToyV3/character/";

int Cfg::get_cls_idx(std::string cls_name)
{
    for (size_t i = 0; i < class_names.size(); i++)
    {
        if (class_names[i] == cls_name)
            return i;
    }
    return -1;
}

//************************Train**************************
int Cfg::epoch = 100;
int Cfg::val_epoch = 1;
float Cfg::lr = 1e-1;
float Cfg::weight_decay = 1e-5;
int Cfg::batch_size = 2;

//************************Predict**************************
double Cfg::score_thr = 0.1;
double Cfg::mask_thr = 0.5;
double Cfg::update_thr = 0.05;
int Cfg::nms_pre = 500;
int Cfg::max_per_img = 100;
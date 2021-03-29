#include <iostream>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "detector.h"
#include "cxxopts.hpp"

using namespace cv;
using namespace std;

void Demo(cv::Mat& img,
        const std::vector<std::vector<int>>& result,
        bool label = true) {

    if (!result.empty()) {
        for(int i = 0; i < result.size(); ++i){
            std::cout << "this is the :" << i<<"box "<< endl;
            std::cout << "this box x is:" << result[i][0] << endl;
            std::cout << "this box y is:" << result[i][1] << endl;
            cv::circle(img, cv::Point(result[i][0], result[i][1]), 5, cv::Scalar(0, 0, 255),-1);
        }
        // auto font_face = cv::FONT_HERSHEY_DUPLEX;
        // auto font_scale = 1.0;
        // int thickness = 1;// int baseline=0;
        // auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
         // cv::rectangle(img,
         //         cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
        //         cv::Point(box.tl().x + s_size.width, box.tl().y),
        //         cv::Scalar(0, 0, 255), -1);
         // cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
        //             font_face , font_scale, cv::Scalar(255, 255, 255), thickness);
    }
    cv::imwrite("./result.jpg", img);
}


int main(int argc, const char* argv[]) {
    cxxopts::Options parser(argv[0], "A LibTorch inference implementation of the yolov5");

    // TODO: add other args
    parser.allow_unrecognised_options().add_options()
            ("weights", "model.torchscript.pt path", cxxopts::value<std::string>())
            ("source", "source", cxxopts::value<std::string>())
            ("gpu", "Enable cuda device or cpu", cxxopts::value<bool>()->default_value("false"))
            ("view-img", "display results", cxxopts::value<bool>()->default_value("false"))
            ("h,help", "Print usage");

    auto opt = parser.parse(argc, argv);

    if (opt.count("help")) {
        std::cout << parser.help() << std::endl;
        exit(0);
    }

    // check if gpu flag is set
    bool is_gpu = opt["gpu"].as<bool>();

    // set device type - CPU/GPU
    torch::DeviceType device_type;
    if (torch::cuda::is_available() && is_gpu) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }

    // load network
    std::string weights = opt["weights"].as<std::string>();
    std::cout << "--------detector-----------:" << std::endl;
    auto detector = Detector(weights, device_type);
    // load input image
    std::string source = opt["source"].as<std::string>();
    std::cout << "source:" << source << std::endl;
    cv::Mat img = cv::imread(source);
    if (img.empty()) {
        std::cerr << "Error loading the image!\n";
        return -1;
    }

    // run once to warm up
    std::cout << "Run once on empty image" << std::endl;
    auto temp_img = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);
    detector.Run(temp_img);
    // inference
    std::vector<std::vector<int>> result = detector.Run(img);
    std::cout << "result" << std::endl;
    // visualize 
    if (opt["view-img"].as<bool>()) {
        Demo(img, result);
    }
    return 0;
}

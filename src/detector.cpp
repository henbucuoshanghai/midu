#include "detector.h"
#include <typeinfo>

Detector::Detector(const std::string& model_path, const torch::DeviceType& device_type) : device_(device_type) {
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        std::cout << "model_path:" << model_path << std::endl;
        module_ = torch::jit::load(model_path);
        std::cout << "--------module_--------" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model!\n";
        std::exit(EXIT_FAILURE);
    }

    half_ = (device_ != torch::kCPU);
    module_.to(device_);

    if (half_) {
        module_.to(torch::kHalf);
    }

    module_.eval();
}


std::vector<std::vector<int>> Detector::Run(const cv::Mat& img) {
    torch::NoGradGuard no_grad;
    std::cout << "----------New Frame----------" << std::endl;

    /*** Pre-process ***/
    auto start = std::chrono::high_resolution_clock::now();
    // keep the original image for visualization purpose
    cv::Mat img_input = img.clone();
    cv::cvtColor(img_input, img_input, cv::COLOR_BGR2RGB);  // BGR -> RGB
    img_input.convertTo(img_input, CV_32FC3, 1.0f / 255.0f);  // normalization 1/255
    auto tensor_img = torch::from_blob(img_input.data, {1, img_input.rows, img_input.cols, img_input.channels()}).to(device_);

    tensor_img = tensor_img.permute({0, 3, 1, 2}).contiguous();  // BHWC -> BCHW (Batch, Channel, Height, Width)

    if (half_) {
        tensor_img = tensor_img.to(torch::kHalf);
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(tensor_img);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // It should be known that it takes longer time at first time
    std::cout << "pre-process takes : " << duration.count() << " ms" << std::endl;

    /*** Inference ***/
    // TODO: add synchronize point
    start = std::chrono::high_resolution_clock::now();

    // inference
    torch::jit::IValue output = module_.forward(inputs);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // It should be known that it takes longer time at first time
    std::cout << "inference takes : " << duration.count() << " ms" << std::endl;

    /*** Post-process ***/

    start = std::chrono::high_resolution_clock::now();
    auto pred_threshold = output.toTuple()->elements()[0].toTensor();
    auto pred_map = output.toTuple()->elements()[1].toTensor();

    auto result = PostProcessing(pred_threshold,pred_map);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // It should be known that it takes longer time at first time
    std::cout << "post-process takes : " << duration.count() << " ms" << std::endl;

    return result;
}

//std::vector<std::vector<float>>
std::vector<std::vector<int>> Detector::PostProcessing(const torch::Tensor& pred_threshold,
          const torch::Tensor& pred_map) {
    //auto biny=torch::relu(pred_map-pred_threshold).dtype(torch::kUInt8));
    torch::Tensor bibibi=torch::relu(pred_map-pred_threshold);
    auto aiaiai=bibibi.to(torch::kBool);
    auto cicici=aiaiai.to(torch::kFloat);//auto cicici=aiaiai.to(torch::kFloat).cpu();
    auto dididi=cicici.squeeze().detach();
    dididi=dididi.mul(255).to(torch::kU8);
    auto output = dididi.to(torch::kCPU);

    cv::Mat result(output.size(0), output.size(1), CV_8UC1);
    std::memcpy((void*)result.data, output.data_ptr(), sizeof(torch::kU8) * output.numel());
    cv::Mat res1,res2,res3;
    //auto dididi=torch::squeeze(cicici).detach().permute({1, 2, 0});     
    cv::connectedComponentsWithStats(result,res1,res2,res3,4,4);
    std::cout<<typeid(res3).name()<<std::endl;
    std::cout<<res3<<std::endl;
    std::vector<std::vector<int>> ooo;
    for (int row = 0; row < res3.rows; row++) {
        std::vector<int> output1;
        for (int col = 0; col < res3.cols; col++) {
			auto pixel = res3.at<double>(row,col); 
            output1.push_back((int)pixel); 
        }
        ooo.push_back(output1);
    }
    return ooo;
}


void Detector::ScaleCoordinates(std::vector<Detection>& data,float pad_w, float pad_h,
                                float scale, const cv::Size& img_shape) {
    auto clip = [](float n, float lower, float upper) {
        return std::max(lower, std::min(n, upper));
    };

    Detection det;
    std::vector<Detection> detections;
    for (auto & i : data) {
        float x1 = (i.bbox.tl().x - pad_w)/scale;  // x padding
        float y1 = (i.bbox.tl().y - pad_h)/scale;  // y padding
        float x2 = (i.bbox.br().x - pad_w)/scale;  // x padding
        float y2 = (i.bbox.br().y - pad_h)/scale;  // y padding

        x1 = clip(x1, 0, img_shape.width);
        y1 = clip(y1, 0, img_shape.height);
        x2 = clip(x2, 0, img_shape.width);
        y2 = clip(y2, 0, img_shape.height);

        i.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));

        std::cout << "tl.x:" << i.bbox.tl().x << " tl.y:" << i.bbox.tl().y << " width:" << i.bbox.width << " height:" << i.bbox.height << std::endl;

    }
}


torch::Tensor Detector::xywh2xyxy(const torch::Tensor& x) {
    auto y = torch::zeros_like(x);
    // convert bounding box format from (center x, center y, width, height) to (x1, y1, x2, y2)
    y.select(1, Det::tl_x) = x.select(1, 0) - x.select(1, 2).div(2);
    y.select(1, Det::tl_y) = x.select(1, 1) - x.select(1, 3).div(2);
    y.select(1, Det::br_x) = x.select(1, 0) + x.select(1, 2).div(2);
    y.select(1, Det::br_y) = x.select(1, 1) + x.select(1, 3).div(2);
    return y;
}


void Detector::Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes,
                                const at::TensorAccessor<float, 2>& det,
                                std::vector<cv::Rect>& offset_box_vec,
                                std::vector<float>& score_vec) {

    for (int i = 0; i < offset_boxes.size(0) ; i++) {
        offset_box_vec.emplace_back(
                cv::Rect(cv::Point(offset_boxes[i][Det::tl_x], offset_boxes[i][Det::tl_y]),
                         cv::Point(offset_boxes[i][Det::br_x], offset_boxes[i][Det::br_y]))
        );
        score_vec.emplace_back(det[i][Det::score]);
    }
}
// src/inference/inference.h
#ifndef YOLO_V12_INFERENCE_H
#define YOLO_V12_INFERENCE_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 如果 Detection 結構體沒有在其他通用頭文件中定義，請保留在這裡
struct Detection {
    cv::Rect bbox;
    float score;
    int class_id;
    std::string class_name;
};

class YOLOv12Inference {
public:
    // 建構函數
    YOLOv12Inference(const std::string& model_path,
                     const std::vector<std::string>& class_names,
                     const Ort::SessionOptions& session_options,
                     float conf_threshold);

    // 解構函數
    ~YOLOv12Inference();

    // 在預處理後的圖像上運行推論
    std::vector<Detection> runInference(const cv::Mat& processed_image);

    // 這些成員變數需要是 public 或提供 getter 函數，以便 main.cpp 訪問
    int64_t _input_height; // 模型期望的輸入高度
    int64_t _input_width;  // 模型期望的輸入寬度

private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions _allocator; // 用於 ONNX Runtime 操作的分配器

    std::vector<std::string> _class_names;
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    float _conf_threshold; // 新增成員變數，用於儲存置信度閾值

    // 輔助函數，用於獲取模型輸入/輸出資訊
    void get_model_info();

};

#endif // YOLO_V12_INFERENCE_H
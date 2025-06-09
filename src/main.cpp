// src/main.cpp
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// 引入所有模塊的頭文件
#include "preprocess/preprocess.h"
#include "inference/inference.h"
#include "postprocess/postprocess.h"
#include "utils/utils.h"

// 定義模型輸入尺寸和閾值
const float CONF_THRESHOLD = 0.25f; // 置信度閾值
const float NMS_THRESHOLD = 0.45f;  // NMS 閾值

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_onnx_model> <path_to_image> [path_to_class_names.names]" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string names_path = (argc > 3) ? argv[3] : "data/coco.names"; // 默認路徑

    // 1. 加載類別名稱
    std::vector<std::string> class_names = loadClassNames(names_path);
    if (class_names.empty()) {
        std::cerr << "Failed to load class names. Exiting." << std::endl;
        return -1;
    }

    // ========================================================================
    // 新增部分：初始化 Ort::SessionOptions
    // ========================================================================
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC); // 設置圖優化級別
    
    // 嘗試啟用 CUDA 執行提供者 (如果你的 ONNX Runtime 支持 GPU 並且你有 CUDA 環境)
    try {
        OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0); // 0 是 GPU device ID
        std::cout << "ONNX Runtime 推論將嘗試使用 GPU (CUDA) 加速。" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "警告: 無法啟用 CUDA 執行提供者，將回退到 CPU 推論。錯誤: " << e.what() << std::endl;
    }
    // ========================================================================

    // 2. 初始化 YOLOv12 推論引擎，現在傳遞 conf_threshold 參數
    YOLOv12Inference yolo_inference(model_path, class_names, session_options, CONF_THRESHOLD);

    // 3. 讀取圖像
    cv::Mat original_image = cv::imread(image_path);
    if (original_image.empty()) {
        std::cerr << "Error: Could not read image: " << image_path << std::endl;
        return -1;
    }
    int original_width = original_image.cols;
    int original_height = original_image.rows;

    std::cout << "Processing image: " << image_path << std::endl;

    Timer timer; // 計時器開始

    // 4. 圖像預處理 (LetterBox + Normalization)
    // 現在使用 yolo_inference._input_width 和 yolo_inference._input_height
    // 這些值是從 ONNX 模型中動態獲取的，確保與模型輸入尺寸一致
    LetterBoxInfo letterbox_info = letterbox(original_image, yolo_inference._input_width, yolo_inference._input_height);

    // ========================================================================
    // 新增部分：印出 LetterBoxInfo 和原始圖像尺寸
    // ========================================================================
    std::cout << "\n--- LetterBoxInfo and Original Image Dimensions ---\n" << std::endl;
    std::cout << "LetterBoxInfo.scale: " << letterbox_info.scale << std::endl;
    std::cout << "LetterBoxInfo.pad_x: " << letterbox_info.pad_x << std::endl;
    std::cout << "LetterBoxInfo.pad_y: " << letterbox_info.pad_y << std::endl;
    std::cout << "Original Image Width: " << original_width << std::endl;
    std::cout << "Original Image Height: " << original_height << std::endl;
    std::cout << "---------------------------------------------------\n" << std::endl;
    // ========================================================================

    cv::Mat processed_input_blob = normalizeAndTranspose(letterbox_info.processed_image);

    // 5. 執行模型推論
    std::vector<Detection> raw_detections = yolo_inference.runInference(processed_input_blob);

    // 6. 後處理 (NMS + 坐標恢復)
    // 篩選置信度 (模型內部可能已篩選，這裡可以再篩選一次)
    std::vector<Detection> filtered_detections;
    for(const auto& det : raw_detections) {
        // 現在 CONF_THRESHOLD 在 YOLOv12Inference::runInference 內部已經篩選過了
        // 所以這裡不需要再次篩選，或者可以調整閾值
        // if (det.score >= CONF_THRESHOLD) {
            filtered_detections.push_back(det);
        // }
    }

    // 執行 NMS
    std::vector<Detection> nms_detections = nonMaximumSuppression(filtered_detections, NMS_THRESHOLD);

    // --- Debugging: Print nms_detections before scaling ---
    std::cout << "\n--- NMS Detections (Before Scaling) ---\n" << std::endl;
    if (nms_detections.empty()) {
        std::cout << "No detections after NMS." << std::endl;
    } else {
        long print_count_nms = 0;
        for (const auto& det : nms_detections) {
            if (print_count_nms >= 5) break;
            std::cout << "Class: " << det.class_name 
                      << ", Score: " << det.score 
                      << ", BBox: [x=" << det.bbox.x 
                      << ", y=" << det.bbox.y 
                      << ", w=" << det.bbox.width 
                      << ", h=" << det.bbox.height << "]\n" << std::endl;
            print_count_nms++;
        }
        if (nms_detections.size() > 5) {
            std::cout << "... (顯示了前 5 個 NMS 檢測。總共有 " << nms_detections.size() << " 個檢測)\n" << std::endl;
        }
    }
    std::cout << "-----------------------------------------\n" << std::endl;
    // ---------------------------------------------------------

    // 將檢測框坐標映射回原始圖像尺寸
    std::vector<Detection> final_detections = scaleDetections(nms_detections,
                                                               letterbox_info,
                                                               original_width,
                                                               original_height);

    // ========================================================================
    // 新增部分：印出後處理後的預測機率和框的位置
    // ========================================================================
    std::cout << "\n--- Final Detections (After Post-processing and Scaling) ---\n" << std::endl;
    if (final_detections.empty()) {
        std::cout << "No detections after post-processing." << std::endl;
    } else {
        // 僅印出前 5 個檢測結果
        long print_count = 0;
        for (const auto& det : final_detections) {
            if (print_count >= 5) break; // 限制為前 5 個
            std::cout << "Class: " << det.class_name 
                      << ", Score: " << det.score 
                      << ", BBox: [x=" << det.bbox.x 
                      << ", y=" << det.bbox.y 
                      << ", w=" << det.bbox.width 
                      << ", h=" << det.bbox.height << "]\n" << std::endl;
            print_count++;
        }
        if (final_detections.size() > 5) {
            std::cout << "... (顯示了前 5 個最終檢測。總共有 " << final_detections.size() << " 個檢測)\n" << std::endl;
        }
    }
    std::cout << "--------------------------------------------------------\n" << std::endl;
    // ========================================================================

    timer.elapsed_ms(); // 計時器結束
    std::cout << "Inference and Post-processing took: " << timer.elapsed_ms() << " ms\n" << std::endl;

    // 7. 在圖像上繪製檢測結果
    cv::Mat drawn_image = original_image.clone(); // 複製一份圖像用於繪製
    drawDetections(drawn_image, final_detections);
    
    // 8. 保存結果
    std::string output_filename = "output/output_detection_result.jpg";
    cv::imwrite(output_filename, drawn_image);
    std::cout << "Detection result saved to " << output_filename << std::endl;

    // 9. 顯示結果 (已註銷)
    cv::imshow("YOLOv12 Detection Result", drawn_image);
    cv::waitKey(0); // 等待按鍵關閉窗口

    return 0;
}
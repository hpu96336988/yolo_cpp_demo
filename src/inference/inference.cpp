// src/inference/inference.cpp
#include "inference.h" // 包含我們自己的頭文件
#include <iostream>
#include <numeric>   // 可能用於某些累積操作，目前程式碼中不直接使用
#include <stdexcept> // 用於拋出標準異常
#include <array>     // 用於 std::array
#include <cmath>     // 用於 expf 函數
#include <algorithm> // 用於 std::sort

// YOLOv12Inference 類的建構函數
YOLOv12Inference::YOLOv12Inference(const std::string& model_path,
                                 const std::vector<std::string>& class_names,
                                 const Ort::SessionOptions& session_options,
                                 float conf_threshold)
    : env(ORT_LOGGING_LEVEL_WARNING, "YOLOv12Inference") // 初始化 ONNX Runtime 環境，設置日誌級別和實例名
    , session(env, model_path.c_str(), session_options) // 初始化 ONNX Session，載入模型
    , _allocator(Ort::AllocatorWithDefaultOptions())    // 初始化 ONNX Runtime 預設分配器
    , _class_names(class_names)                         // 儲存類別名稱
    , _input_height(0)                                  // 初始化模型輸入高度為 0
    , _input_width(0)                                   // 初始化模型輸入寬度為 0
    , _conf_threshold(conf_threshold)                     // 初始化置信度閾值
{
    // 呼叫輔助函數來獲取模型的輸入/輸出節點名稱和維度等信息
    get_model_info();
    std::cout << "YOLOv12Inference 已用模型初始化: " << model_path << std::endl;
}

// YOLOv12Inference 類的解構函數
YOLOv12Inference::~YOLOv12Inference() {
    // ONNX Runtime 會自動管理 session、env 和 allocator 的生命週期，
    // 所以這裡不需要手動釋放資源。
}

// 獲取模型輸入/輸出資訊的輔助函數
void YOLOv12Inference::get_model_info() {
    // 獲取模型輸入節點的數量
    size_t num_input_nodes = session.GetInputCount();
    if (num_input_nodes == 0) {
        throw std::runtime_error("模型沒有輸入節點。");
    }
    // 對於 YOLO 模型，通常只有一個輸入節點，我們處理第一個
    auto input_name_allocated = session.GetInputNameAllocated(0, _allocator);
    input_node_names.push_back(input_name_allocated.get());

    // 獲取輸入張量的類型和形狀資訊
    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_dims = tensor_info.GetShape();

    // 假設輸入張量是 NCHW (Batch, Channels, Height, Width) 格式
    // 檢查維度是否符合 4D 張量的預期，並且 Batch size 應為 1
    if (input_dims.size() != 4 || input_dims[0] != 1) {
        std::string error_msg = "意外的輸入張量形狀。預期 4 個維度且批次大小為 1。當前形狀：[";
        for (size_t i = 0; i < input_dims.size(); ++i) {
            error_msg += std::to_string(input_dims[i]);
            if (i < input_dims.size() - 1) error_msg += ", ";
        }
        error_msg += "]。";
        throw std::runtime_error(error_msg);
    }

    // 儲存模型期望的輸入高度和寬度
    _input_height = input_dims[2]; // Height
    _input_width = input_dims[3];  // Width

    // 獲取模型輸出節點的數量
    size_t num_output_nodes = session.GetOutputCount();
    if (num_output_nodes == 0) {
        throw std::runtime_error("模型沒有輸出節點。");
    }
    // 對於 YOLO 模型，通常只有一個輸出節點，我們處理第一個
    auto output_name_allocated = session.GetOutputNameAllocated(0, _allocator);
    output_node_names.push_back(output_name_allocated.get());

    // 檢查是否成功獲取了輸入和輸出節點名稱
    if (input_node_names.empty() || output_node_names.empty()) {
        throw std::runtime_error("未能獲取模型輸入/輸出名稱。模型可能格式不正確或為空。");
    }

    std::cout << "模型輸入節點: " << input_node_names[0] << std::endl;
    std::cout << "模型輸出節點: " << output_node_names[0] << std::endl;
    std::cout << "模型期望輸入尺寸 (H, W): " << _input_height << ", " << _input_width << std::endl;
}

// Sigmoid 函式實現
// float YOLOv12Inference::sigmoid(float x) const {
//     return 1.0f / (1.0f + expf(-x));
// }

// 在預處理後的圖像上運行推論
std::vector<Detection> YOLOv12Inference::runInference(const cv::Mat& processed_image) {
    // 1. 準備輸入張量
    // processed_image 應該已經是 NCHW (1, C, H, W) 格式的 float32 blob
    // 我們使用從模型資訊中獲取的 input_height 和 input_width 來確保形狀一致性
    std::vector<int64_t> input_tensor_shape = {
        1,                            // Batch size (通常為 1)
        3,                            // Channels (通常為 3 for RGB/BGR)
        _input_height,                // Height (從模型獲取)
        _input_width                  // Width (從模型獲取)
    };

    // 確保 processed_image 的數據類型是 CV_32F (float)
    if (processed_image.type() != CV_32F) {
        throw std::runtime_error("處理後的圖像必須是 CV_32F 類型。");
    }

    // 手動計算張量的總元素數量 (N * C * H * W)
    // 這是解決之前 "negative value in shape" 錯誤的關鍵
    size_t input_tensor_size = 1;
    for (int64_t dim : input_tensor_shape) {
        if (dim <= 0) {
            // 這個檢查應該在 get_model_info 或 preprocess 階段就捕獲，但作為額外安全檢查保留
            throw std::runtime_error("輸入張量形狀包含非正維度 (內部檢查)。");
        }
        input_tensor_size *= dim;
    }

    // 創建 CPU 記憶體資訊對象
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // 創建 ONNX Runtime 輸入張量
    // 使用 processed_image.data 作為數據指針，以及手動計算的元素總數和正確的形狀
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        (float*)processed_image.data,  // 使用 cv::Mat 的數據指針
        input_tensor_size,             // 總元素數 (N * C * H * W)
        input_tensor_shape.data(),     // 輸入張量形狀
        input_tensor_shape.size()      // 輸入張量形狀的維度數 (通常是 4)
    );

    // 2. 執行推論
    std::vector<Ort::Value> output_tensors;
    try {
        // 將 std::string 的向量轉換為 const char* 的陣列，以符合 Ort::Session::Run 的簽名
        std::vector<const char*> input_node_names_c_str;
        for (const auto& name : input_node_names) {
            input_node_names_c_str.push_back(name.c_str());
        }

        std::vector<const char*> output_node_names_c_str;
        for (const auto& name : output_node_names) {
            output_node_names_c_str.push_back(name.c_str());
        }

        output_tensors = session.Run(Ort::RunOptions{nullptr},
                                     input_node_names_c_str.data(), // 輸入節點名稱陣列
                                     &input_tensor,                 // 輸入張量
                                     1,                             // 輸入張量數量
                                     output_node_names_c_str.data(),// 輸出節點名稱陣列
                                     output_node_names_c_str.size()); // 輸出節點數量
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime 推論失敗: " << e.what() << std::endl;
        return {}; // 返回空檢測結果
    }

    // 3. 解析輸出張量 (YOLOv12 的輸出結構需要根據實際模型而定)
    // 假設 YOLOv12 的輸出是一個 [1, num_detections, 5 + num_classes] 的張量
    // 其中 5 通常是 [x_center, y_center, width, height, confidence_score]
    // 後續是每個類別的置信度

    std::vector<Detection> detections;
    if (output_tensors.empty()) {
        std::cerr << "推論結果為空，沒有輸出張量。" << std::endl;
        return {};
    }

    const float* output_data = output_tensors[0].GetTensorData<float>();
    Ort::TensorTypeAndShapeInfo output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_info.GetShape();

    // 輸出張量形狀通常是 [1, num_boxes, 5 + num_classes]
    std::cout << "onnx輸出格式: ["; // 新增的輸出標籤
    for (size_t i = 0; i < output_shape.size(); ++i) {
        std::cout << output_shape[i];
        if (i < output_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl; // 結束形狀輸出

    if (output_shape.size() != 3 || output_shape[0] != 1) {
        std::cerr << "意外的輸出張量形狀。預期 3 個維度且批次大小為 1，得到 "
                  << output_shape.size() << " 個維度，批次大小為 " << output_shape[0] << std::endl;
        // 如果輸出形狀與預期不符，打印詳細信息並返回空檢測
        std::cerr << "實際輸出形狀：[";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cerr << output_shape[i];
            if (i < output_shape.size() - 1) std::cerr << ", ";
        }
        std::cerr << "]" << std::endl;
        return {};
    }

    long num_attributes = output_shape[1]; // 修正：現在 output_shape[1] 是屬性數量 (84)
    long num_boxes = output_shape[2];      // 修正：現在 output_shape[2] 是邊界框數量 (8400)

    // ========================================================================
    // 新增部分：直接印出模型原始輸出數據 (僅限前 num_attributes * 5 個浮點數)
    // ========================================================================
    // std::cout << "\n--- Raw Model Output (First " << num_attributes * 5 << " values) ---\n" << std::endl;
    // long print_limit = std::min((long)num_attributes * 5, num_attributes * num_boxes); // 最多印出前5個框的數據
    // for (long i = 0; i < print_limit; ++i) {
    //     std::cout << output_data[i] << " ";
    //     if ((i + 1) % num_attributes == 0) { // 每印完一個框的數據就換行
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << "--------------------------------------------------\n" << std::endl;
    // ========================================================================

    for (long i = 0; i < num_boxes; ++i) {
        // 從原始輸出數據中獲取 x1, y1, x2, y2 (YOLOv8 的 xyxy 格式)
        // 由於輸出格式是 [1, attributes, num_boxes]，數據是按列主序排列的
        float x1 = output_data[0 * num_boxes + i];
        float y1 = output_data[1 * num_boxes + i];
        float x2 = output_data[2 * num_boxes + i];
        float y2 = output_data[3 * num_boxes + i];

        // 從 xyxy 計算寬度和高度
        float width = x2 - x1;
        float height = y2 - y1;

        // 獲取所有類別分數，並找到最大分數及其對應的類別 ID
        float max_score = -1.0f;
        int class_id = -1;
        // 類別分數從第 4 個屬性開始 (索引 4)
        for (long j = 0; j < _class_names.size(); ++j) { // j 是類別索引
            float score = output_data[(4 + j) * num_boxes + i];
            // 不應用 sigmoid，直接使用原始分數
            if (score > max_score) {
                max_score = score;
                class_id = j;
            }
        }

        if (max_score >= _conf_threshold) {
            Detection det;
            // 使用 x1, y1, width, height 構造 cv::Rect2f
            det.bbox = cv::Rect2f(x1, y1, width, height);
            det.score = max_score;
            if (class_id >= 0 && class_id < _class_names.size()) {
                det.class_name = _class_names[class_id];
            } else {
                det.class_name = "Unknown";
            }
            detections.push_back(det);
        }
    }
    return detections;
}
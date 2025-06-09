#include "preprocess.h"
#include <iostream>

// 實現 LetterBox 圖像調整
LetterBoxInfo letterbox(const cv::Mat& image, int target_width, int target_height) {
    LetterBoxInfo info;
    info.processed_image = cv::Mat(); // 初始化為空

    int img_width = image.cols;
    int img_height = image.rows;

    // 計算縮放比例
    float ratio_w = (float)target_width / img_width;
    float ratio_h = (float)target_height / img_height;
    float scale = std::min(ratio_w, ratio_h); // 取最小比例以確保圖像完整放入

    int new_width = static_cast<int>(img_width * scale);
    int new_height = static_cast<int>(img_height * scale);

    // 調整圖像大小
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

    // 計算填充值
    int pad_w = target_width - new_width;
    int pad_h = target_height - new_height;

    // 填充黑色邊框 (128, 128, 128 為 YOLOv5/v8 常用填充色)
    // 左右各填充 pad_w/2，上下各填充 pad_h/2
    cv::copyMakeBorder(resized_image, info.processed_image,
                       pad_h / 2, pad_h - pad_h / 2, // top, bottom
                       pad_w / 2, pad_w - pad_w / 2, // left, right
                       cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));

    info.scale = scale;
    info.pad_x = pad_w / 2;
    info.pad_y = pad_h / 2;

    std::cout << "LetterBox: original (" << img_width << "," << img_height
              << "), target (" << target_width << "," << target_height
              << "), scaled (" << new_width << "," << new_height
              << "), padding (" << pad_w << "," << pad_h << "), scale: " << scale << std::endl;

    return info;
}

// 實現圖像數據正規化和通道轉置 (HWC -> CHW)
// 假定模型輸入是 float32，範圍 0-1
cv::Mat normalizeAndTranspose(const cv::Mat& image) {
    cv::Mat float_image;
    // 將圖像數據類型轉換為 float32
    image.convertTo(float_image, CV_32FC3, 1.0 / 255.0); // 歸一化到 0-1 範圍

    // 由於 ONNX Runtime 通常期望 NCHW (Batch, Channel, Height, Width) 格式
    // OpenCV 的 Mat 默認是 NHWC (Height, Width, Channel)
    // 我們需要將 HWC 轉換為 CHW，然後再提供給 ONNX Runtime 的張量
    // ONNX Runtime 會自動處理 batch 維度

    // 方法1: 手動分離通道並合併 (較為直觀但可能稍慢)
    // std::vector<cv::Mat> channels;
    // cv::split(float_image, channels);
    // cv::Mat blob_chw;
    // cv::vconcat(channels[0], channels[1], blob_chw);
    // cv::vconcat(blob_chw, channels[2], blob_chw);

    // 方法2: 使用 dnn::blobFromImage (推薦，但需要確保其內部行為符合期望)
    // 如果模型輸入是 RGB，而 OpenCV 讀取是 BGR，這裡可能需要 BGR2RGB 轉換
    // 對於YOLOv12，通常輸入是RGB
    cv::Mat rgb_image;
    cv::cvtColor(float_image, rgb_image, cv::COLOR_BGR2RGB); // 通常需要 BGR2RGB

    cv::Mat blob;
    // blobFromImage 預設會進行 1/255.0 縮放，這裡我們已手動進行，所以 scale_factor 設為 1.0
    // dnn::blobFromImage 的輸出是 NCHW (batch, channel, height, width) 格式
    // 這裡我們只處理單張圖片，所以 batch=1
    cv::dnn::blobFromImage(rgb_image, blob, 1.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
    // ====================================================================
    // 添加以下代碼來檢查 blob 的維度
    // ====================================================================
    std::cout << "Blob dimensions (N, C, H, W): "
              << blob.size[0] << ", "  // N (batch size)
              << blob.size[1] << ", "  // C (channels)
              << blob.size[2] << ", "  // H (height)
              << blob.size[3] << std::endl; // W (width)

    if (blob.empty()) {
        std::cerr << "Error: blobFromImage returned empty blob!" << std::endl;
    }
    // 檢查是否有負值
    if (blob.size[0] <= 0 || blob.size[1] <= 0 || blob.size[2] <= 0 || blob.size[3] <= 0) {
        std::cerr << "Error: Blob has non-positive dimensions! N=" << blob.size[0]
                  << ", C=" << blob.size[1] << ", H=" << blob.size[2] << ", W=" << blob.size[3] << std::endl;
    }
    // ====================================================================

    return blob;
}

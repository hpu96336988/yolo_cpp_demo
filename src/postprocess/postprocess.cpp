#include "postprocess.h"
#include <algorithm> // For std::sort, std::max, std::min

// 實現非極大值抑制 (NMS)
std::vector<Detection> nonMaximumSuppression(std::vector<Detection>& detections, float nms_threshold) {
    if (detections.empty()) {
        return {};
    }

    // 按置信度分數降序排序
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }

            // 計算 IoU (Intersection over Union)
            cv::Rect bbox1 = detections[i].bbox;
            cv::Rect bbox2 = detections[j].bbox;

            cv::Rect intersection = bbox1 & bbox2; // 交集
            float iou = (float)intersection.area() / (bbox1.area() + bbox2.area() - intersection.area());

            if (iou > nms_threshold) {
                suppressed[j] = true; // 抑制重疊度高的框
            }
        }
    }
    return result;
}

// 實現將檢測框坐標從模型輸入尺寸映射回原始圖像尺寸
std::vector<Detection> scaleDetections(const std::vector<Detection>& detections,
                                       const LetterBoxInfo& letterbox_info,
                                       int original_img_width,
                                       int original_img_height) {
    std::vector<Detection> scaled_detections;
    for (const auto& det : detections) {
        Detection scaled_det = det;

        // Debug: Print LetterBoxInfo and original image dimensions inside scaleDetections
        // std::cout << "  [scaleDetections Debug] LetterBoxInfo.scale: " << letterbox_info.scale 
        //           << ", pad_x: " << letterbox_info.pad_x 
        //           << ", pad_y: " << letterbox_info.pad_y << std::endl;
        // std::cout << "  [scaleDetections Debug] Original Image W: " << original_img_width 
        //           << ", H: " << original_img_height << std::endl;

        // Debug: Print input det.bbox to scaleDetections
        // std::cout << "  [scaleDetections Debug] Input BBox: [x=" << det.bbox.x 
        //           << ", y=" << det.bbox.y 
        //           << ", w=" << det.bbox.width 
        //           << ", h=" << det.bbox.height << "]" << std::endl;

        // 將坐標從填充後的模型尺寸恢復到原始縮放比例
        // 坐標 = (模型輸出坐標 - 填充量) / 縮放比例
        scaled_det.bbox.x = static_cast<int>((det.bbox.x - letterbox_info.pad_x) / letterbox_info.scale);
        scaled_det.bbox.y = static_cast<int>((det.bbox.y - letterbox_info.pad_y) / letterbox_info.scale);
        scaled_det.bbox.width = static_cast<int>(det.bbox.width / letterbox_info.scale);
        scaled_det.bbox.height = static_cast<int>(det.bbox.height / letterbox_info.scale);

        // Debug: Print scaled_det.bbox before clamping
        // std::cout << "  [scaleDetections Debug] Before Clamping BBox: [x=" << scaled_det.bbox.x 
        //           << ", y=" << scaled_det.bbox.y 
        //           << ", w=" << scaled_det.bbox.width 
        //           << ", h=" << scaled_det.bbox.height << "]" << std::endl;

        // 確保坐標在原始圖像範圍內
        scaled_det.bbox.x = std::max(0, scaled_det.bbox.x);
        scaled_det.bbox.y = std::max(0, scaled_det.bbox.y);
        scaled_det.bbox.width = std::min(original_img_width - scaled_det.bbox.x, scaled_det.bbox.width);
        scaled_det.bbox.height = std::min(original_img_height - scaled_det.bbox.y, scaled_det.bbox.height);

        // Debug: Print scaled_det.bbox after clamping
        // std::cout << "  [scaleDetections Debug] After Clamping BBox: [x=" << scaled_det.bbox.x 
        //           << ", y=" << scaled_det.bbox.y 
        //           << ", w=" << scaled_det.bbox.width 
        //           << ", h=" << scaled_det.bbox.height << "]" << std::endl;

        scaled_detections.push_back(scaled_det);
    }
    return scaled_detections;
}

// 在圖像上繪製檢測結果
void drawDetections(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        // 繪製邊界框
        cv::rectangle(image, det.bbox, cv::Scalar(0, 255, 0), 2); // 綠色框

        // 繪製類別標籤和置信度
        std::string label = det.class_name + ": " + cv::format("%.2f", det.score);
        int baseLine;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int x = det.bbox.x;
        int y = det.bbox.y - 10 > 0 ? det.bbox.y - 10 : 0; // 確保文字在圖像內部

        cv::rectangle(image, cv::Rect(x, y, label_size.width, label_size.height + baseLine),
                      cv::Scalar(0, 255, 0), cv::FILLED); // 填充背景
        cv::putText(image, label, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1); // 黑色文字
    }
}

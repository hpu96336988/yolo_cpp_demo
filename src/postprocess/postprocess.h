#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "../inference/inference.h" // 引入 Detection 結構體
#include "../preprocess/preprocess.h" // 引入 LetterBoxInfo 結構體

// 執行非極大值抑制 (NMS)
std::vector<Detection> nonMaximumSuppression(std::vector<Detection>& detections,
                                             float nms_threshold);

// 將檢測框坐標從模型輸入尺寸映射回原始圖像尺寸
std::vector<Detection> scaleDetections(const std::vector<Detection>& detections,
                                       const LetterBoxInfo& letterbox_info,
                                       int original_img_width,
                                       int original_img_height);

// 在圖像上繪製檢測結果
void drawDetections(cv::Mat& image, const std::vector<Detection>& detections);

#endif // POSTPROCESS_H

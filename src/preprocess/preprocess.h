#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <vector>

// 結構體：用於存儲預處理後的圖像數據和必要的縮放/填充信息
struct LetterBoxInfo {
cv::Mat processed_image; // 預處理後的圖像
float scale; // 圖像縮放比例
int pad_x; // 填充的水平像素數
int pad_y; // 填充的垂直像素數
// 可選：用於後續恢復到原始圖像坐標的變換矩陣
// cv::Mat transform_matrix;
};
// 函數宣告：執行 LetterBox 操作，保持長寬比縮放並填充
// 這個函數將原始圖像調整大小並填充，使其適應模型的輸入尺寸
LetterBoxInfo letterbox(const cv::Mat& image, int target_width, int target_height);

// 函數宣告：執行圖像數據正規化和通道轉置 (HWC -> CHW)
// 這個函數將圖像像素值歸一化到 0-1 範圍，並將圖像從 HWC 格式轉換為 NCHW (對於單張圖片 N=1)
cv::Mat normalizeAndTranspose(const cv::Mat& image);
#endif // PREPROCESS_H
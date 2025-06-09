#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <chrono> // For timing

// 讀取 YOLO 模型所需的類別名稱文件 (例如 coco.names)
std::vector<std::string> loadClassNames(const std::string& names_path);

// 簡單的計時器類 (可選)
class Timer {
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

#endif // UTILS_H
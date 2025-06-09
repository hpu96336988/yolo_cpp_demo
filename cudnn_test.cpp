// cudnn_test.cpp
#include <iostream>
#include <cudnn.h> // 包含 cuDNN 的頭文件
#include <cstdlib> // 為了使用 exit()

#define CHECK_CUDNN(status) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() { // <-- 這裡是 main 函數的開始！
    cudnnHandle_t cudnn;
    // 嘗試創建一個 cuDNN 句柄
    // 這是 cuDNN 函式庫最基本的調用之一
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // 獲取 cuDNN 版本
    size_t version = cudnnGetVersion();
    std::cout << "Successfully created cuDNN handle." << std::endl;
    std::cout << "cuDNN version: " << version / 1000 << "." << (version / 100) % 10 << "." << (version % 100) << std::endl;

    // 清理句柄
    CHECK_CUDNN(cudnnDestroy(cudnn));

    std::cout << "cuDNN test passed!" << std::endl;
    return 0;
} // <-- 這裡是 main 函數的結束！
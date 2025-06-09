cmake_minimum_required(VERSION 3.10)
project(YOLOv12Demo CXX)

# 設置 C++ 標準
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# --- ONNX Runtime 配置 ---
# 手動設置 ONNX Runtime 的安裝路徑
# 這應該指向您解壓縮 onnxruntime-linux-x64-gpu-1.22.0.tgz 後的頂層資料夾
set(ONNXRUNTIME_DIR "/usr/local/onnxruntime-linux-x64-gpu-1.22.0")

# 添加 ONNX Runtime 的頭文件路徑
# 確保指向包含 onnxruntime_c_api.h 等文件的 include 資料夾
include_directories(${ONNXRUNTIME_DIR}/include)

# 添加 ONNX Runtime 的庫文件路徑
# 確保指向包含 libonnxruntime.so 等文件的 lib 資料夾
link_directories(${ONNXRUNTIME_DIR}/lib)
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# 遍歷所有源文件
file(GLOB_RECURSE SRC_FILES "src/*.cpp" "src/*/*.cpp") # 遞歸查找所有 .cpp 文件

# 添加可執行文件
add_executable(yolov12_demo ${SRC_FILES})

# 鏈接庫
target_link_libraries(yolov12_demo
    ${OpenCV_LIBS}
    # 鏈接 ONNX Runtime 的主庫
    onnxruntime
    # 如果您的 ONNX Runtime 預編譯包是 GPU 版本，通常需要鏈接 CUDA 和 TensorRT 相關的提供者庫
    # 這些庫名可能因 ONNX Runtime 版本和編譯選項而異，請檢查 ONNXRUNTIME_DIR/lib 目錄
    # 例如：
    # onnxruntime_providers_cuda # 如果你使用了 CUDA
    # onnxruntime_providers_tensorrt # 如果你使用了 TensorRT
    # onnxruntime_providers_shared # 有些版本會有這個，檢查一下
)

# 為了方便 CMake 找到其他非標準路徑下的庫，也可以考慮添加
# set(CMAKE_INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
# set(CMAKE_BUILD_RPATH "${ONNXRUNTIME_DIR}/lib")

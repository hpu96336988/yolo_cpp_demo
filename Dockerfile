# 使用 NVIDIA CUDA 官方映像作為基礎映像。
# 這個映像包含了 CUDA 工具包、cuDNN 和 Ubuntu 22.04 作業系統，
# 非常適合需要 GPU 加速的深度學習應用。
# 如果您不需要 GPU 支援，可以考慮使用更輕量級的 Ubuntu 或 Debian 基礎映像，
# 例如 FROM ubuntu:22.04。
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# 設定容器內部的工作目錄。
# 所有後續的命令（如 COPY, RUN, CMD）如果沒有指定絕對路徑，
# 都會在這個目錄下執行。這有助於組織容器內部的檔案。
WORKDIR /app

# 安裝專案所需的建置工具和函式庫。
# DEBIAN_FRONTEND=noninteractive 讓 apt-get 不會問問題。
    # TZ=Asia/Taipei 預設時區（你可改成你想要的時區）。
    # 把 tzdata 放進安裝清單，這樣時區套件會自動設定。
    # 這樣可以避免在安裝過程中出現互動式提示，沒有這麼做會在build時卡住
        #    => [3/5] RUN apt-get update && apt-get install -y --no-install-recommends build-essential 6314.3s
        # => => # questions will narrow this down by presenting a list of cities, representing
        # => => # the time zones in which they are located.
        # => => # 1. Africa 4. Australia 7. Atlantic 10. Pacific 13. Legacy
        # => => # 2. America 5. Arctic 8. Europe 11. US
        # => => # 3. Antarctica 6. Asia 9. Indian 12. Etc
        # => => # Geographic area:

# apt-get update：更新套件列表。
# apt-get install -y --no-install-recommends：安裝指定的套件。
#   -y：自動回答 "yes" 以避免互動式確認。
#   --no-install-recommends：只安裝必要的依賴，不安裝推薦的依賴，以減小映像檔大小。
# build-essential：包含了編譯 C++ 程式所需的工具（如 gcc, g++）。
# cmake：用於建置您的 C++ 專案。
# libopencv-dev：OpenCV 函式庫的開發檔案，YOLO C++ 專案通常會依賴它。
# git：如果您的專案需要從 Git 倉庫克隆其他依賴，則需要它。
# && rm -rf /var/lib/apt/lists/*：在安裝完成後清除 apt 緩存，以減小最終映像檔的大小。
ENV CONTAINER_TIMEZONE=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN DEBIAN_FRONTEND=noninteractive TZ=Asia/Taipei \
    apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    build-essential \
    cmake \
    libopencv-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 將主機（您的電腦）上當前目錄下的所有檔案和資料夾，
# 複製到容器內部的工作目錄 /app 中。
# 這個 . (點) 表示當前目錄。第一個 . 是主機的當前目錄，第二個 . 是容器內的當前工作目錄。
# 安裝 ONNX Runtime C++ 預編譯版（以 1.17.0 為例，可依需求調整版本）
# 安裝 ONNX Runtime GPU 版
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-gpu-1.17.0.tgz && \
    tar -xzf onnxruntime-linux-x64-gpu-1.17.0.tgz && \
    cp -r onnxruntime-linux-x64-gpu-1.17.0/include/* /usr/local/include/ && \
    cp -r onnxruntime-linux-x64-gpu-1.17.0/lib/* /usr/local/lib/ && \
    ldconfig && \
    rm -rf onnxruntime-linux-x64-gpu-1.17.0*


COPY . .

# 建立建置目錄並進行編譯。
# RUN mkdir build：在 /app 目錄下建立一個名為 build 的資料夾。
# && cd build：進入 build 資料夾。
# && cmake ..：運行 CMake，它會讀取 CMakeLists.txt 來生成 Makefile。
#   .. 表示 CMakeLists.txt 在上一級目錄（即 /app）。
# && make -j$(nproc)：使用 make 命令編譯專案。
#   -j$(nproc)：這個選項會告訴 make 使用您系統所有可用的 CPU 核心進行並行編譯，以加快速度。
RUN mkdir -p build && cd build && \
    cmake .. || cat CMakeFiles/CMakeError.log && \
    make -j$(nproc) || cat CMakeFiles/CMakeError.log

# 設定環境變數 (可選)。
# 如果您的應用程式需要特定的環境變數來找到模型、配置檔案或其他資源，
# 可以在這裡設定。例如，如果您的可執行檔在 /app/build 且需要加入 PATH。
# ENV PATH="/app/build:${PATH}"

# 定義容器啟動時執行的預設命令。
# 這是當您運行 Docker 容器時，容器會自動執行的命令。
# 根據您專案中實際存在的 ONNX 模型、類別名稱檔案和測試圖片進行精確配置。
# 假設您的可執行檔名為 `yolo_cpp_demo` 且在 `build` 目錄下。
# **請注意：如果您的可執行檔名稱不是 `yolo_cpp_demo`，請修改此處。**
CMD ["/app/build/yolov12_demo", "models/yolov8n.onnx", "images/000000000001.jpg", "data/coco.names"]
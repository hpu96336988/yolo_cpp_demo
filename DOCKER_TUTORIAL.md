## Docker 部署

[TOC]

### Docker 部署教學：YOLO C++ ONNX 專案 (新手專用)

#### 簡介

歡迎來到 Docker 的世界！Docker 是一個非常強大的工具，它可以幫助您將應用程式（例如您的 YOLO C++ ONNX 專案）及其所有運行所需的依賴項（函式庫、環境變數等）打包到一個獨立的、輕量級的單元中，這個單元我們稱之為「容器 (Container)」。

想像一下，您的應用程式就像一個特別的植物，它需要特定的土壤、陽光和水分才能健康生長。Docker 就像是一個標準化的「花盆」，無論您把這個花盆放在哪裡（您的電腦、另一台伺服器），只要花盆裡有植物所需的所有東西，它就能正常運行。

**為什麼要用 Docker？**

*   **環境一致性：** 解決「在我機器上可以跑，在你機器上卻不行」的問題。確保您的應用程式在任何地方都能以相同的方式運行。
*   **隔離性：** 您的應用程式及其依賴項都在容器內，不會影響到您主機系統上安裝的其他軟體。
*   **可重複性：** 只要您有 `Dockerfile`，就可以在任何兼容 Docker 的機器上，輕鬆地重新建立相同的運行環境。
*   **部署簡單：** 將應用程式部署到伺服器上時，只需要運行一個 Docker 命令即可。

#### 前置條件

在我們開始之前，請確保您的電腦上已經安裝了以下必要的軟體：

1.  **Docker Desktop (適用於 Windows 和 macOS) 或 Docker Engine (適用於 Linux)：**
    *   這是 Docker 的核心組件，它允許您在電腦上建置和運行 Docker 容器。
    *   您可以前往 Docker 官方網站下載並按照指示安裝：[https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
    *   **重要提示：** 安裝完成後，請確保 Docker 服務正在運行。在 Docker Desktop 中，您通常會看到應用程式圖示顯示為綠色或「Running」。
2.  **NVIDIA Container Toolkit (可選，但如果您想使用 GPU 運行 YOLO 專案，則為必需)：**
    *   如果您的電腦配備了 NVIDIA 顯示卡，並且您希望利用 GPU 的強大計算能力來加速 YOLO 的推論過程，那麼您需要安裝這個工具。它允許 Docker 容器訪問您主機上的 NVIDIA GPU。
    *   安裝說明：[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
    *   如果您沒有 NVIDIA GPU，或者不打算使用 GPU 運行，則可以跳過此步驟。您的 YOLO 專案將會使用 CPU 進行推論。

#### 步驟 1：建立 `Dockerfile`

`Dockerfile` 是一個非常重要的文本檔案。它就像是 Docker 的「藍圖」，裡面包含了 Docker 應該如何一步步地建置您的應用程式映像檔的所有指令。

請在您的專案根目錄下（也就是您目前所在的位置，`CMakeLists.txt`、`src/`、`models/` 等資料夾所在的同一層）建立一個名為 `Dockerfile` 的新檔案。**請注意檔案名稱必須是 `Dockerfile`，沒有副檔名。**

然後，請將以下內容複製貼上到您剛建立的 `Dockerfile` 中：

```Dockerfile
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

# 設定容器的時區。
# 這兩行是確保容器內部時區設定為 Asia/Taipei，避免在一些套件安裝時可能遇到的互動式提示，
# 尤其是 tzdata 套件。
# ln -snf ...: 建立一個符號連結，將系統的時區資訊指向指定的時區。
# echo ... > ...: 將時區名稱寫入 /etc/timezone 檔案。
ENV CONTAINER_TIMEZONE=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# apt-get update：更新套件列表。
# apt-get install -y --no-install-recommends：安裝指定的套件。
#   -y：自動回答 "yes" 以避免互動式確認。
#   --no-install-recommends：只安裝必要的依賴，不安裝推薦的依賴，以減小映像檔大小。
# tzdata: 時區數據包，配合上面的 ENV 和 RUN 命令來避免時區設置的互動提示。
# build-essential：包含了編譯 C++ 程式所需的工具（如 gcc, g++）。
# cmake：用於建置您的 C++ 專案。
# libopencv-dev：OpenCV 函式庫的開發檔案，YOLO C++ 專案通常會依賴它。
# git：如果您的專案需要從 Git 倉庫克隆其他依賴，則需要它。
# wget：用於從網路下載檔案，例如 ONNX Runtime。
# && rm -rf /var/lib/apt/lists/*：在安裝完成後清除 apt 緩存，以減小最終映像檔的大小。
RUN DEBIAN_FRONTEND=noninteractive TZ=Asia/Taipei \
    apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    build-essential \
    cmake \
    libopencv-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安裝 ONNX Runtime C++ 預編譯版（以 1.17.0 為例，可依需求調整版本）。
# 這個區塊負責下載 ONNX Runtime 的 GPU 版本，解壓縮，
# 並將其頭文件和函式庫複製到系統標準路徑，以便編譯器和連結器能夠找到。
# wget ...: 從指定 URL 下載 ONNX Runtime GPU 版的壓縮包。
# tar -xzf ...: 解壓縮下載的 .tgz 檔案。
# cp -r .../include/* /usr/local/include/: 將 ONNX Runtime 的所有頭文件複製到系統的標準包含路徑。
# cp -r .../lib/* /usr/local/lib/: 將 ONNX Runtime 的所有函式庫文件複製到系統的標準函式庫路徑。
# ldconfig: 更新動態連結函式庫的緩存，確保系統能找到新複製的函式庫。
# rm -rf ...: 清理下載的壓縮包和解壓縮後的文件夾，以減小映像檔大小。
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-gpu-1.17.0.tgz && \
    tar -xzf onnxruntime-linux-x64-gpu-1.17.0.tgz && \
    cp -r onnxruntime-linux-x64-gpu-1.17.0/include/* /usr/local/include/ && \
    cp -r onnxruntime-linux-x64-gpu-1.17.0/lib/* /usr/local/lib/ && \
    ldconfig && \
    rm -rf onnxruntime-linux-x64-gpu-1.17.0*

# 將主機（您的電腦）上當前目錄下的所有檔案和資料夾，
# 複製到容器內部的工作目錄 /app 中。
# 這個 . (點) 表示當前目錄。第一個 . 是主機的當前目錄，第二個 . 是容器內的當前工作目錄。
COPY . .

# 建立建置目錄並進行編譯。
# RUN mkdir -p build：在 /app 目錄下建立一個名為 build 的資料夾。
# && cd build：進入 build 資料夾。
# && cmake ..：運行 CMake，它會讀取 CMakeLists.txt 來生成 Makefile。
#   .. 表示 CMakeLists.txt 在上一級目錄（即 /app）。
# cmake ... || cat CMakeFiles/CMakeError.log: 如果 cmake 失敗，會將錯誤日誌輸出到終端，方便除錯。
# && make -j$(nproc)：使用 make 命令編譯專案。
#   -j$(nproc)：這個選項會告訴 make 使用您系統所有可用的 CPU 核心進行並行編譯，以加快速度。
# make ... || cat CMakeFiles/CMakeError.log: 如果 make 失敗，也會將錯誤日誌輸出到終端，方便除錯。
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
# 假設您的可執行檔名為 `yolov12_demo` 且在 `build` 目錄下。
# **請注意：如果您的可執行檔名稱不是 `yolov12_demo`，請修改此處。**
# 此處的 models/yolov8n.onnx 和 images/000000000001.jpg 是範例路徑，請確保其在您的專案中存在且路徑正確。
CMD ["/app/build/yolov12_demo", "models/yolov8n.onnx", "images/000000000001.jpg", "data/coco.names"]
```

**`Dockerfile` 內容解說：**

我已經在上面的程式碼中加入了詳細的註解，解釋了每一行指令的作用。請仔細閱讀這些註解，它們對於理解 Dockerfile 的工作原理非常重要。

#### 步驟 2：建置 Docker 映像檔 (Build Docker Image)

當您有了 `Dockerfile` 之後，下一步就是使用它來建置一個 Docker 映像檔。映像檔是容器的「範本」，它包含了運行應用程式所需的所有代碼、函式庫、工具和環境。

請在您的專案根目錄（`Dockerfile` 所在的位置）打開您的終端機（命令提示字元或 PowerShell 在 Windows 上，Terminal 在 macOS 或 Linux 上）。

然後，執行以下命令：

```bash
docker build -t yolov12-demo .
```

**命令解釋：**

*   `docker build`: 這是 Docker 命令，用於根據 `Dockerfile` 建置映像檔。
*   `-t yolo-cpp-app`: 這個選項用於為您的映像檔「命名」和「標記」。
    *   `yolo-cpp-app` 是您給這個映像檔取的名字，您可以改成任何您喜歡的名字（建議使用小寫字母和連字號）。
    *   Docker 會自動為它加上 `:latest` 的標籤，表示這是這個名字的最新版本。您可以指定其他標籤，例如 `-t yolo-cpp-app:v1.0`。
*   `.`: 這個點表示 `Dockerfile` 位於當前的工作目錄中。如果您的 `Dockerfile` 在其他位置，您需要指定該路徑。

**執行這個命令後會發生什麼？**

當您在終端機中執行上述命令後，Docker 將會開始建置映像檔。您會看到終端機上顯示出類似以下的進度信息。這些是 Docker 建置過程中會顯示的日誌和狀態更新（可能會有很多行，因為它在下載基礎映像、安裝軟體和編譯您的專案）：

```
[+] Building 0.0s (0/0)
... (這裡會顯示 Docker 正在執行每個步驟的詳細信息，例如下載、安裝、複製文件等)
Step 1/7 : FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
...
Successfully built <image_id>
Successfully tagged yolo-cpp-app:latest
```

當您看到 `Successfully built <image_id>` (其中 `<image_id>` 是一串亂碼，代表映像檔的唯一 ID) 和 `Successfully tagged yolo-cpp-app:latest` 時，這表示您的 Docker 映像檔已經成功建置完成。這個過程可能需要一些時間，特別是第一次下載基礎映像和編譯專案時。

#### 步驟 3：運行 Docker 容器 (Run Docker Container)

建置好映像檔後，您就可以從這個映像檔中啟動一個或多個「容器」。容器是映像檔的運行實例。

**根據您是否使用 NVIDIA GPU，運行命令會有所不同：**

**情況 A：如果您有 NVIDIA GPU 並已安裝 NVIDIA Container Toolkit (推薦，以便利用 GPU 加速)：**

```bash
docker run --gpus all -it --rm \\
    -v "$(pwd)/images:/app/images" \\
    -v "$(pwd)/models:/app/models" \\
    -v "$(pwd)/data:/app/data" \\
    -v "$(pwd)/output:/app/output" \\
    yolo-cpp-app
```

**情況 B：如果您沒有 NVIDIA GPU 或不想使用 GPU (使用 CPU 運行)：**

```bash
docker run -it --rm \\
    -v "$(pwd)/images:/app/images" \\
    -v "$(pwd)/models:/app/models" \\
    -v "$(pwd)/data:/app/data" \\
    -v "$(pwd)/output:/app/output" \\
    yolov12-demo 
```

**命令解釋（非常重要，請仔細閱讀）：**

*   `docker run`: 這是 Docker 命令，用於從映像檔啟動並運行一個新的容器。
*   `--gpus all` (僅限 GPU 情況):
    *   這個選項是 `nvidia-container-toolkit` 提供的，它會告訴 Docker 容器，讓它可以訪問您主機上**所有可用**的 NVIDIA GPU。
    *   如果您只想指定特定的 GPU，例如 ID 為 `0` 的 GPU，可以使用 `--gpus device=0`。
*   `-it`: 這兩個是常用的選項組合：
    *   `-i` (`--interactive`): 保持標準輸入 (stdin) 開啟，即使沒有連接到容器。這使得您可以在容器內進行互動。
    *   `-t` (`--tty`): 分配一個偽終端機 (pseudo-TTY)。這使得您可以在終端機中看到容器的輸出，並能像在正常的終端機中一樣與容器互動。
*   `--rm`: 這個選項表示當容器停止（例如，您的 YOLO 程式運行結束）時，Docker 會自動刪除這個容器。這有助於保持您的系統清潔，避免留下大量的停止容器。
*   `-v <主機路徑>:<容器路徑>` (`--volume`): 這是**掛載卷 (Volume Mount)** 的關鍵選項。它允許您將主機（您的電腦）上的某個資料夾，映射到容器內部的某個資料夾。
    *   `"$(pwd)/images"`: 這部分代表您**主機**上 `images` 資料夾的絕對路徑。
        *   `$(pwd)` 是一個 Bash/Zsh/PowerShell 的語法，它會自動解析為您當前終端機所在的目錄的絕對路徑。所以，如果您在專案根目錄運行這個命令，它就會是您專案目錄下的 `images` 資料夾。
        *   **如果您在 Windows 的命令提示字元 (CMD) 中，`$(pwd)` 無法直接使用。** 您需要手動將其替換為實際的絕對路徑，例如：
            `C:\\Users\\YourUser\\YourProject\\images`
            請記得將反斜線 `\` 替換為正斜線 `/`，或者使用雙反斜線 `\\`，例如 `C:/Users/YourUser/YourProject/images`。
    *   `/app/images`: 這部分代表容器內部的路徑。由於您在 `Dockerfile` 中將專案複製到了 `/app`，所以容器內的 `images` 資料夾應該是 `/app/images`。
    *   **掛載卷的作用：**
        *   **輸入數據：** 透過將主機上的 `images` 資料夾掛載到容器的 `/app/images`，您的 YOLO 程式在容器內部運行時，可以讀取主機 `images` 資料夾中的圖片。您只需要將新的圖片放入主機上的 `images` 資料夾，容器就可以處理它，而不需要重新建置映像檔。
        *   **輸出結果：** 透過將主機上的 `output` 資料夾掛載到容器的 `/app/output`，您的 YOLO 程式在容器內部生成的所有結果檔案（例如帶有檢測框的圖片）都會直接保存到您主機上的 `output` 資料夾中。這樣您就可以方便地在主機上查看結果。
        *   同樣的原理適用於 `models` 和 `data` 資料夾，確保您的程式可以訪問模型權重、配置檔案和類別名稱檔案。
*   `yolov12-demo`: 這是您在步驟 2 中成功建置的 Docker 映像檔的名稱。

執行上述 `docker run` 命令後，Docker 容器將會啟動，並自動執行您在 `Dockerfile` 中 `CMD` 指令定義的預設命令（也就是您的 `yolo_cpp_demo` 程式）。您應該會在終端機中看到程式的輸出，例如處理進度、日誌信息等。

**替換圖片指令**
```
docker run --gpus all --rm yolov12-demo /app/build/yolov12_demo models/yolov8n.onnx images/000000021079.jpg data/coco.names  
```

**docker運行但輸出結果到本機putput中**

```
docker run --gpus all --rm -v $(pwd)/output:/app/output yolov12-demo /app/build/yolov12_demo models/yolov8n.onnx images/000000000001.jpg data/coco.names
```
#### 步驟 4：驗證結果

當您的 Docker 容器運行結束後（通常當 `yolo_cpp_demo` 程式完成其任務並退出時，容器也會停止），請執行以下操作來驗證結果：

1.  **檢查 `output` 資料夾：**
    *   回到您的專案根目錄（主機上）。
    *   檢查 `output` 資料夾。如果一切順利，您應該會看到 `yolo_cpp_demo` 程式處理後的圖片或其他結果檔案，例如 `output_detection_result.jpg` (這是在 `Dockerfile` 中 `CMD` 命令範例會生成的檔案)。

如果這些檔案存在並且內容正確，那麼恭喜您，您已經成功地在 Docker 容器中運行了您的 YOLO C++ ONNX 專案！

#### 步驟 5 (可選)：進入容器內部進行調試

有時候，您可能希望在容器運行時進入其內部，執行一些命令，檢查檔案系統，或者進行調試。這對於理解容器環境或排除故障非常有用。

1.  **查找容器 ID 或名稱：**
    *   在新的終端機視窗中，執行以下命令來查看所有正在運行或最近停止的容器：
        ```bash
        docker ps -a
        ```
    *   您會看到類似以下的輸出：
        ```
        CONTAINER ID   IMAGE           COMMAND                  CREATED          STATUS                       PORTS     NAMES
        a1b2c3d4e5f6   yolo-cpp-app    "/bin/bash -c '/app/…"   5 minutes ago    Exited (0) 2 minutes ago               amazing_container
        ```
    *   記下 `CONTAINER ID` (例如 `a1b2c3d4e5f6`) 或 `NAMES` (例如 `amazing_container`)。
        *   如果您的容器已經停止（`STATUS` 顯示 `Exited`），您需要先重新運行它（回到步驟 3），因為 `docker exec` 只能對運行中的容器操作。
2.  **進入容器：**
    *   使用您剛剛找到的容器 ID 或名稱，執行以下命令：
        ```bash
        docker exec -it <容器ID或容器名稱> bash
        ```
        例如：
        ```bash
        docker exec -it a1b2c3d4e5f6 bash
        # 或者
        docker exec -it amazing_container bash
        ```
    *   執行此命令後，您將會進入容器內部的 Bash Shell。您會看到終端機提示符變成了容器內部的提示符（例如 `root@a1b2c3d4e5f6:/app#`）。
    *   現在您可以在容器內部執行各種 Linux 命令，例如 `ls` (列出檔案)、`pwd` (顯示當前路徑)、`cd` (切換目錄) 等，就像在一個獨立的 Linux 系統中一樣。
    *   您可以檢查 `/app` 目錄下的內容，查看您的程式和數據檔案是否都在預期的位置。
3.  **退出容器：**
    *   當您完成調試後，只需在容器內部的終端機中輸入 `exit`，然後按下 Enter 鍵即可返回到主機的終端機。

#### 步驟 6 (可選)：清理 Docker 資源

當您不再需要某些 Docker 容器或映像檔時，為了釋放磁碟空間並保持系統整潔，您可以將它們刪除。

1.  **停止並刪除正在運行的容器 (如果有的話)：**
    *   如果您之前運行了 `--rm` 選項，容器在停止後會自動刪除，這一步可能就不需要了。
    *   如果您想手動停止並刪除所有基於 `yolo-cpp-app` 映像檔運行的容器：
        ```bash
        docker stop $(docker ps -a -q --filter ancestor=yolo-cpp-app)
        docker rm $(docker ps -a -q --filter ancestor=yolo-cpp-app)
        ```
        *   `docker ps -a -q --filter ancestor=yolo-cpp-app`: 這個命令會查找所有基於 `yolo-cpp-app` 映像檔（無論是運行中還是已停止）的容器 ID。
        *   `docker stop` 和 `docker rm` 會對這些 ID 執行停止和刪除操作。
2.  **刪除 Docker 映像檔：**
    *   當您確定不再需要 `yolo-cpp-app` 映像檔時，可以將其刪除：
        ```bash
        docker rmi yolo-cpp-app
        ```
        *   `docker rmi`: 這是用於刪除 Docker 映像檔的命令。
        *   請注意，如果您有基於此映像檔的容器仍在運行，Docker 會阻止您刪除映像檔。您需要先停止並刪除所有相關容器。 
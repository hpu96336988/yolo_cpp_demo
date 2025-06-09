#include "utils.h"
#include <fstream>
#include <iostream>

// 實現讀取類別名稱文件
std::vector<std::string> loadClassNames(const std::string& names_path) {
    std::vector<std::string> class_names;
    std::ifstream ifs(names_path);
    if (!ifs.is_open()) {
        std::cerr << "Error: Could not open class names file: " << names_path << std::endl;
        return class_names;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        class_names.push_back(line);
    }
    ifs.close();
    std::cout << "Loaded " << class_names.size() << " class names from " << names_path << std::endl;
    return class_names;
}

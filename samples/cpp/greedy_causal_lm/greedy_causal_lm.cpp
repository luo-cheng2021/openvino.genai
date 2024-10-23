// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include <chrono>

int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\"");

    std::string model_path = argv[1];
    std::string prompt = argv[2];
    std::string device = "CPU";  // GPU can be used as well

    ov::genai::LLMPipeline pipe(model_path, device);
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    config.ignore_eos = true;
    std::cout << "output len,first run cost(seconds), second run cost(seconds)\n";
    for (size_t i = 256; i <= 512; i += 64) {
        config.min_new_tokens = i;
        config.max_new_tokens = i;
        auto beg = std::chrono::high_resolution_clock::now();
        std::string result = pipe.generate(prompt, config);
        auto end = std::chrono::high_resolution_clock::now();
        auto cost0 = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();

        beg = std::chrono::high_resolution_clock::now();
        pipe.generate(prompt, config);
        end = std::chrono::high_resolution_clock::now();
        auto cost1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
        std::cout << i << ", " << cost0 << ", " << cost1 << "\n";
        std::cout << result << std::endl;
    }
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}

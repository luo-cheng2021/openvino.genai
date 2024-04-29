// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"

#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include "model_config.hpp"
#include "device_config.hpp"

void apply_paged_attention_transformations(std::shared_ptr<ov::Model> model, const ModelConfig& model_config,
    const DeviceConfig& device_config, const SchedulerConfig& scheduler_config) {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::SDPAToPagedAttention>();
    manager.run_passes(model);

    const ov::ParameterVector& parameters = model->get_parameters();
    for (size_t decoder_layer_id = 0; decoder_layer_id < model_config.get_num_layers(); ++decoder_layer_id) {
        parameters[2 + 2 * decoder_layer_id]->set_element_type(device_config.get_cache_precision());
        parameters[2 + 2 * decoder_layer_id + 1]->set_element_type(device_config.get_cache_precision());
        parameters[2 + 2 * decoder_layer_id]->set_partial_shape(device_config.get_key_cache_shape());
        parameters[2 + 2 * decoder_layer_id + 1]->set_partial_shape(device_config.get_value_cache_shape());
    }

    if (scheduler_config.dynamic_split_fuse) {
        bool has_subsequence_lens = false;
        for (auto&& parameter : parameters) {
            auto&& names = parameter->output(0).get_names();
            if (names.count("subsequence_lens")) {
                has_subsequence_lens = true;
                break;
            }
        }
        if (!has_subsequence_lens) {
            auto subsequence_lens = std::make_shared<ov::opset8::Parameter>(ov::element::i64, ov::PartialShape{-1});
            subsequence_lens->set_friendly_name("subsequence_lens");
            subsequence_lens->get_output_tensor(0).set_names({"subsequence_lens"});
            model->add_parameters({subsequence_lens});

            for (const auto& op : model->get_ops()) {
                if (op->get_type_name() == std::string("PagedAttentionExtension")) {
                    auto inputs = op->input_values();
                    inputs.push_back(subsequence_lens);
                    auto new_op = op->clone_with_new_inputs(inputs);
                    ov::replace_node(op, new_op);
                }
            }
        }
    }

    model->validate_nodes_and_infer_types();
}

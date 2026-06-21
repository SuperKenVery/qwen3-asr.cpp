#include "audio_encoder.h"
#include "timing.h"

#include <cmath>
#include <cstring>
#include <algorithm>

#define QWEN3_ASR_MAX_NODES 4096

namespace qwen3_asr {

static void compute_sinusoidal_pe(float * pe, int n_ctx, int d_model) {
    const int half_dim = d_model / 2;
    for (int pos = 0; pos < n_ctx; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            float div_term = expf(-logf(10000.0f) * i / (half_dim - 1));
            float angle = pos * div_term;
            pe[pos * d_model + i] = sinf(angle);
            pe[pos * d_model + half_dim + i] = cosf(angle);
        }
    }
}

static void add_positional_embedding(const audio_encoder_model & model,
                                     std::vector<float> & data,
                                     int64_t n_ctx,
                                     int64_t n_state) {
    if (model.pos_embd_w &&
        model.pos_embd_w->type == GGML_TYPE_F32 &&
        model.pos_embd_w->ne[0] == n_state &&
        model.pos_embd_w->ne[1] >= n_ctx &&
        model.pos_embd_w->data) {
        const float * pos = static_cast<const float *>(model.pos_embd_w->data);
        for (int64_t t = 0; t < n_ctx; ++t) {
            for (int64_t i = 0; i < n_state; ++i) {
                data[t * n_state + i] += pos[t * n_state + i];
            }
        }
        return;
    }

    std::vector<float> pe(n_ctx * n_state);
    compute_sinusoidal_pe(pe.data(), n_ctx, n_state);
    for (int64_t i = 0; i < n_ctx * n_state; ++i) {
        data[i] += pe[i];
    }
}

AudioEncoder::AudioEncoder() = default;

AudioEncoder::~AudioEncoder() {
    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend_gpu) {
        ggml_backend_free(state_.backend_gpu);
        state_.backend_gpu = nullptr;
    }
    if (state_.backend_cpu) {
        ggml_backend_free(state_.backend_cpu);
        state_.backend_cpu = nullptr;
    }
    free_model(model_);
}

bool AudioEncoder::load_model(const std::string & model_path) {
    GGUFLoader loader;
    if (!loader.load(model_path, model_)) {
        error_msg_ = loader.get_error();
        return false;
    }
    
    state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!state_.backend_cpu) {
        error_msg_ = "Failed to initialize CPU backend";
        return false;
    }

    state_.backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);

    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> backend_bufts;

    if (state_.backend_gpu) {
        backends.push_back(state_.backend_gpu);
        backend_bufts.push_back(ggml_backend_get_default_buffer_type(state_.backend_gpu));
    }

    backends.push_back(state_.backend_cpu);
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_get_default_buffer_type(state_.backend_cpu);
    if (state_.backend_gpu) {
        ggml_backend_dev_t gpu_dev = ggml_backend_get_device(state_.backend_gpu);
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(gpu_dev);
        if (host_buft) cpu_buft = host_buft;
    }
    backend_bufts.push_back(cpu_buft);

    state_.sched = ggml_backend_sched_new(backends.data(), backend_bufts.data(), backends.size(), QWEN3_ASR_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }
    
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_ASR_MAX_NODES + ggml_graph_overhead());
    
    return true;
}

struct ggml_cgraph * AudioEncoder::build_graph_conv(int n_frames) {
    const auto & hp = model_.hparams;
    const int n_mel = hp.n_mel_bins;
    const int conv_ch = hp.conv_channels;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    
    struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, n_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);
    
    struct ggml_tensor * mel_4d = ggml_reshape_4d(ctx0, mel, n_frames, n_mel, 1, 1);
    
    struct ggml_tensor * cur = ggml_conv_2d(ctx0, model_.conv2d1_w, mel_4d, 2, 2, 1, 1, 1, 1);
    if (model_.conv2d1_b) {
        struct ggml_tensor * bias = ggml_reshape_4d(ctx0, model_.conv2d1_b, 1, 1, conv_ch, 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    ggml_set_name(cur, "after_conv1_pre_gelu");
    ggml_set_output(cur);
    cur = ggml_gelu(ctx0, cur);
    ggml_set_name(cur, "after_conv1");
    ggml_set_output(cur);
    
    cur = ggml_conv_2d(ctx0, model_.conv2d2_w, cur, 2, 2, 1, 1, 1, 1);
    if (model_.conv2d2_b) {
        struct ggml_tensor * bias = ggml_reshape_4d(ctx0, model_.conv2d2_b, 1, 1, conv_ch, 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    cur = ggml_gelu(ctx0, cur);
    
    cur = ggml_conv_2d(ctx0, model_.conv2d3_w, cur, 2, 2, 1, 1, 1, 1);
    if (model_.conv2d3_b) {
        struct ggml_tensor * bias = ggml_reshape_4d(ctx0, model_.conv2d3_b, 1, 1, conv_ch, 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    cur = ggml_gelu(ctx0, cur);
    
    ggml_set_name(cur, "after_conv3");
    ggml_set_output(cur);
    
    int64_t out_w = cur->ne[0];
    int64_t out_h = cur->ne[1];
    int64_t out_c = cur->ne[2];
    int64_t seq_len = out_w;
    int64_t feat_dim = out_c * out_h;
    
    cur = ggml_reshape_3d(ctx0, cur, out_w, out_h * out_c, 1);
    cur = ggml_transpose(ctx0, cur);
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_2d(ctx0, cur, feat_dim, seq_len);
    
    ggml_set_name(cur, "before_conv_out");
    ggml_set_output(cur);
    
    if (model_.conv_out_w) {
        cur = ggml_mul_mat(ctx0, model_.conv_out_w, cur);
    }
    
    ggml_set_name(cur, "embd_conv");
    ggml_set_output(cur);
    state_.embd_conv = cur;
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

struct ggml_cgraph * AudioEncoder::build_graph_encoder(int n_ctx) {
    const auto & hp = model_.hparams;
    const int n_state = hp.d_model;
    const int n_head = hp.n_attention_heads;
    const int n_layer = hp.n_encoder_layers;
    const int n_state_head = n_state / n_head;
    const float eps = hp.layer_norm_eps;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_ASR_MAX_NODES, false);
    
    struct ggml_tensor * cur = ggml_view_tensor(ctx0, state_.embd_conv);
    
    const float KQscale = 1.0f / sqrtf(float(n_state_head));
    
    struct ggml_tensor * inpL = cur;
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        {
            cur = ggml_norm(ctx0, inpL, eps);
            if (layer.attn_norm_w) {
                cur = ggml_mul(ctx0, cur, layer.attn_norm_w);
            }
            if (layer.attn_norm_b) {
                cur = ggml_add(ctx0, cur, layer.attn_norm_b);
            }
        }
        
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q_w, cur);
            if (layer.attn_q_b) {
                Qcur = ggml_add(ctx0, Qcur, layer.attn_q_b);
            }
            
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k_w, cur);
            if (layer.attn_k_b) {
                Kcur = ggml_add(ctx0, Kcur, layer.attn_k_b);
            }
            
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v_w, cur);
            if (layer.attn_v_b) {
                Vcur = ggml_add(ctx0, Vcur, layer.attn_v_b);
            }
            
            struct ggml_tensor * Q = ggml_permute(ctx0,
                ggml_reshape_3d(ctx0, Qcur, n_state_head, n_head, n_ctx),
                0, 2, 1, 3);
            
            struct ggml_tensor * K = ggml_permute(ctx0,
                ggml_reshape_3d(ctx0, Kcur, n_state_head, n_head, n_ctx),
                0, 2, 1, 3);
            
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_ext(ctx0, KQ, nullptr, KQscale, 0.0f);
            
            struct ggml_tensor * V = ggml_cont(ctx0, ggml_permute(ctx0,
                ggml_reshape_3d(ctx0, Vcur, n_state_head, n_head, n_ctx),
                1, 2, 0, 3));
            
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            
            cur = ggml_cont_2d(ctx0, KQV_merged, n_state, n_ctx);
        }
        
        {
            cur = ggml_mul_mat(ctx0, layer.attn_out_w, cur);
            if (layer.attn_out_b) {
                cur = ggml_add(ctx0, cur, layer.attn_out_b);
            }
        }
        
        cur = ggml_add(ctx0, cur, inpL);
        
        struct ggml_tensor * inpFF = cur;
        
        {
            {
                cur = ggml_norm(ctx0, inpFF, eps);
                if (layer.ffn_norm_w) {
                    cur = ggml_mul(ctx0, cur, layer.ffn_norm_w);
                }
                if (layer.ffn_norm_b) {
                    cur = ggml_add(ctx0, cur, layer.ffn_norm_b);
                }
            }
            
            cur = ggml_mul_mat(ctx0, layer.ffn_up_w, cur);
            if (layer.ffn_up_b) {
                cur = ggml_add(ctx0, cur, layer.ffn_up_b);
            }
            
            cur = ggml_gelu(ctx0, cur);
            
            cur = ggml_mul_mat(ctx0, layer.ffn_down_w, cur);
            if (layer.ffn_down_b) {
                cur = ggml_add(ctx0, cur, layer.ffn_down_b);
            }
        }
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;
    
    ggml_set_name(cur, "embd_enc");
    ggml_set_output(cur);
    state_.embd_enc = cur;
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

bool AudioEncoder::compute_graph(struct ggml_cgraph * graph) {
    if (!ggml_backend_sched_alloc_graph(state_.sched, graph)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
    
    if (ggml_backend_sched_graph_compute(state_.sched, graph) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    ggml_backend_sched_reset(state_.sched);
    return true;
}

static int compute_chunk_output_length(int chunk_len) {
    int len = chunk_len;
    len = (len - 1) / 2 + 1;
    len = (len - 1) / 2 + 1;
    len = (len - 1) / 2 + 1;
    return len;
}

static std::vector<float> build_window_attention_mask(int32_t n_ctx, int32_t window_size) {
    if (window_size <= 0) {
        window_size = n_ctx;
    }

    std::vector<float> mask((size_t)n_ctx * n_ctx, -INFINITY);
    for (int32_t start = 0; start < n_ctx; start += window_size) {
        int32_t end = std::min(start + window_size, n_ctx);
        for (int32_t r = start; r < end; ++r) {
            for (int32_t c = start; c < end; ++c) {
                mask[(size_t)r * n_ctx + c] = 0.0f;
            }
        }
    }
    return mask;
}

bool AudioEncoder::encode(const float * mel_data, int n_mel, int n_frames, 
                          std::vector<float> & output) {
    QWEN3_TIMER("audio_encoding.total");
    
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    if (n_mel != model_.hparams.n_mel_bins) {
        error_msg_ = "Mel bins mismatch: expected " + std::to_string(model_.hparams.n_mel_bins) + 
                     ", got " + std::to_string(n_mel);
        return false;
    }
    
    const int n_window = 50;
    const int chunk_size = n_window * 2;
    const int n_state = model_.hparams.d_model;
    
    int n_chunks = (n_frames + chunk_size - 1) / chunk_size;
    
    std::vector<int> chunk_lengths(n_chunks);
    std::vector<int> chunk_output_lengths(n_chunks);
    int total_output_frames = 0;
    
    for (int i = 0; i < n_chunks; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, n_frames);
        chunk_lengths[i] = end - start;
        chunk_output_lengths[i] = compute_chunk_output_length(chunk_lengths[i]);
        total_output_frames += chunk_output_lengths[i];
    }
    
    std::vector<float> all_conv_outputs;
    all_conv_outputs.reserve(total_output_frames * n_state);
    
    for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
        QWEN3_TIMER("audio_encoding.conv_chunk");
        int chunk_start = chunk_idx * chunk_size;
        int chunk_len = chunk_lengths[chunk_idx];
        int chunk_out_len = chunk_output_lengths[chunk_idx];
        
        struct ggml_cgraph * gf_conv = build_graph_conv(chunk_len);
        
        if (!ggml_backend_sched_alloc_graph(state_.sched, gf_conv)) {
            error_msg_ = "Failed to allocate conv graph for chunk " + std::to_string(chunk_idx);
            return false;
        }
        
        struct ggml_tensor * mel_tensor = ggml_graph_get_tensor(gf_conv, "mel");
        if (!mel_tensor) {
            error_msg_ = "Failed to find mel tensor";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
        
        std::vector<float> chunk_mel(n_mel * chunk_len);
        for (int m = 0; m < n_mel; ++m) {
            for (int f = 0; f < chunk_len; ++f) {
                chunk_mel[f + m * chunk_len] = mel_data[m * n_frames + chunk_start + f];
            }
        }
        
        ggml_backend_tensor_set(mel_tensor, chunk_mel.data(), 0, n_mel * chunk_len * sizeof(float));
        
        if (ggml_backend_sched_graph_compute(state_.sched, gf_conv) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute conv graph for chunk " + std::to_string(chunk_idx);
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
        
        struct ggml_tensor * embd_conv = ggml_graph_get_tensor(gf_conv, "embd_conv");
        if (!embd_conv) {
            error_msg_ = "Failed to find embd_conv tensor";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
        
        int64_t out_ctx = embd_conv->ne[1];
        int64_t out_state = embd_conv->ne[0];
        
        if (out_ctx != chunk_out_len) {
            fprintf(stderr, "WARNING: Expected %d output frames, got %lld\n", chunk_out_len, (long long)out_ctx);
        }
        
        std::vector<float> chunk_output(out_ctx * out_state);
        ggml_backend_tensor_get(embd_conv, chunk_output.data(), 0, out_ctx * out_state * sizeof(float));
        
        add_positional_embedding(model_, chunk_output, out_ctx, out_state);
        
        all_conv_outputs.insert(all_conv_outputs.end(), chunk_output.begin(), chunk_output.end());
        
        ggml_backend_sched_reset(state_.sched);
    }
    
    int64_t n_ctx = total_output_frames;
    const int conv_chunk_output = compute_chunk_output_length(chunk_size);
    const int window_aftercnn = conv_chunk_output * std::max(1, model_.hparams.n_window_infer / chunk_size);
    std::vector<float> attn_mask = build_window_attention_mask(n_ctx, window_aftercnn);
    
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_ASR_MAX_NODES + ggml_graph_overhead());
    
    struct ggml_init_params enc_params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * enc_ctx = ggml_init(enc_params);
    struct ggml_cgraph * gf_enc = ggml_new_graph_custom(enc_ctx, QWEN3_ASR_MAX_NODES, false);
    
    const auto & hp = model_.hparams;
    const int n_head = hp.n_attention_heads;
    const int n_layer = hp.n_encoder_layers;
    const int n_state_head = n_state / n_head;
    const float eps = hp.layer_norm_eps;
    const float KQscale = 1.0f / sqrtf(float(n_state_head));
    
    struct ggml_tensor * inpL = ggml_new_tensor_2d(enc_ctx, GGML_TYPE_F32, n_state, n_ctx);
    ggml_set_name(inpL, "enc_input");
    ggml_set_input(inpL);

    struct ggml_tensor * mask_t = ggml_new_tensor_2d(enc_ctx, GGML_TYPE_F32, n_ctx, n_ctx);
    ggml_set_name(mask_t, "attn_mask");
    ggml_set_input(mask_t);
    
    struct ggml_tensor * cur = inpL;
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        {
            cur = ggml_norm(enc_ctx, inpL, eps);
            if (layer.attn_norm_w) {
                cur = ggml_mul(enc_ctx, cur, layer.attn_norm_w);
            }
            if (layer.attn_norm_b) {
                cur = ggml_add(enc_ctx, cur, layer.attn_norm_b);
            }
        }
        
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(enc_ctx, layer.attn_q_w, cur);
            if (layer.attn_q_b) {
                Qcur = ggml_add(enc_ctx, Qcur, layer.attn_q_b);
            }
            
            struct ggml_tensor * Kcur = ggml_mul_mat(enc_ctx, layer.attn_k_w, cur);
            if (layer.attn_k_b) {
                Kcur = ggml_add(enc_ctx, Kcur, layer.attn_k_b);
            }
            
            struct ggml_tensor * Vcur = ggml_mul_mat(enc_ctx, layer.attn_v_w, cur);
            if (layer.attn_v_b) {
                Vcur = ggml_add(enc_ctx, Vcur, layer.attn_v_b);
            }
            
            struct ggml_tensor * Q = ggml_permute(enc_ctx,
                ggml_reshape_3d(enc_ctx, Qcur, n_state_head, n_head, n_ctx),
                0, 2, 1, 3);
            
            struct ggml_tensor * K = ggml_permute(enc_ctx,
                ggml_reshape_3d(enc_ctx, Kcur, n_state_head, n_head, n_ctx),
                0, 2, 1, 3);
            
            struct ggml_tensor * KQ = ggml_mul_mat(enc_ctx, K, Q);
            
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_ext(enc_ctx, KQ, mask_t, KQscale, 0.0f);
            
            struct ggml_tensor * V = ggml_cont(enc_ctx, ggml_permute(enc_ctx,
                ggml_reshape_3d(enc_ctx, Vcur, n_state_head, n_head, n_ctx),
                1, 2, 0, 3));
            
            struct ggml_tensor * KQV = ggml_mul_mat(enc_ctx, V, KQ_soft_max);
            
            struct ggml_tensor * KQV_merged = ggml_permute(enc_ctx, KQV, 0, 2, 1, 3);
            
            cur = ggml_cont_2d(enc_ctx, KQV_merged, n_state, n_ctx);
        }
        
        {
            cur = ggml_mul_mat(enc_ctx, layer.attn_out_w, cur);
            if (layer.attn_out_b) {
                cur = ggml_add(enc_ctx, cur, layer.attn_out_b);
            }
        }
        
        cur = ggml_add(enc_ctx, cur, inpL);
        
        struct ggml_tensor * inpFF = cur;
        
        {
            {
                cur = ggml_norm(enc_ctx, inpFF, eps);
                if (layer.ffn_norm_w) {
                    cur = ggml_mul(enc_ctx, cur, layer.ffn_norm_w);
                }
                if (layer.ffn_norm_b) {
                    cur = ggml_add(enc_ctx, cur, layer.ffn_norm_b);
                }
            }
            
            cur = ggml_mul_mat(enc_ctx, layer.ffn_up_w, cur);
            if (layer.ffn_up_b) {
                cur = ggml_add(enc_ctx, cur, layer.ffn_up_b);
            }
            
            cur = ggml_gelu(enc_ctx, cur);
            
            cur = ggml_mul_mat(enc_ctx, layer.ffn_down_w, cur);
            if (layer.ffn_down_b) {
                cur = ggml_add(enc_ctx, cur, layer.ffn_down_b);
            }
        }
        
        inpL = ggml_add(enc_ctx, cur, inpFF);
    }
    
    cur = inpL;
    
    if (model_.ln_post_w) {
        cur = ggml_norm(enc_ctx, cur, eps);
        cur = ggml_mul(enc_ctx, cur, model_.ln_post_w);
        if (model_.ln_post_b) {
            cur = ggml_add(enc_ctx, cur, model_.ln_post_b);
        }
    }
    
    if (model_.proj1_w) {
        cur = ggml_mul_mat(enc_ctx, model_.proj1_w, cur);
        if (model_.proj1_b) {
            cur = ggml_add(enc_ctx, cur, model_.proj1_b);
        }
        cur = ggml_gelu(enc_ctx, cur);
    }
    
    if (model_.proj2_w) {
        cur = ggml_mul_mat(enc_ctx, model_.proj2_w, cur);
        if (model_.proj2_b) {
            cur = ggml_add(enc_ctx, cur, model_.proj2_b);
        }
    }
    
    ggml_set_name(cur, "embd_enc");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf_enc, cur);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf_enc)) {
        error_msg_ = "Failed to allocate encoder graph";
        ggml_free(enc_ctx);
        return false;
    }
    
    struct ggml_tensor * enc_input = ggml_graph_get_tensor(gf_enc, "enc_input");
    if (!enc_input) {
        error_msg_ = "Failed to find enc_input tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(enc_ctx);
        return false;
    }
    
    ggml_backend_tensor_set(enc_input, all_conv_outputs.data(), 0, n_ctx * n_state * sizeof(float));

    struct ggml_tensor * mask_input = ggml_graph_get_tensor(gf_enc, "attn_mask");
    if (!mask_input) {
        error_msg_ = "Failed to find attn_mask tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(enc_ctx);
        return false;
    }
    
    ggml_backend_tensor_set(mask_input, attn_mask.data(), 0, attn_mask.size() * sizeof(float));
    
    {
        QWEN3_TIMER("audio_encoding.transformer");
        if (ggml_backend_sched_graph_compute(state_.sched, gf_enc) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute encoder graph";
            ggml_backend_sched_reset(state_.sched);
            ggml_free(enc_ctx);
            return false;
        }
    }
    
    struct ggml_tensor * embd_enc = ggml_graph_get_tensor(gf_enc, "embd_enc");
    if (!embd_enc) {
        error_msg_ = "Failed to find embd_enc tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(enc_ctx);
        return false;
    }
    
    int64_t out_n_ctx = embd_enc->ne[1];
    int64_t out_n_state = embd_enc->ne[0];
    
    output.resize(out_n_ctx * out_n_state);
    ggml_backend_tensor_get(embd_enc, output.data(), 0, out_n_ctx * out_n_state * sizeof(float));
    
    ggml_backend_sched_reset(state_.sched);
    ggml_free(enc_ctx);
    
    return true;
}

bool AudioEncoder::encode_no_chunk(const float * mel_data, int n_mel, int n_frames,
                                    std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    if (n_mel != model_.hparams.n_mel_bins) {
        error_msg_ = "Mel bins mismatch";
        return false;
    }
    
    const int n_state = model_.hparams.d_model;
    
    struct ggml_cgraph * gf_conv = build_graph_conv(n_frames);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf_conv)) {
        error_msg_ = "Failed to allocate conv graph";
        return false;
    }
    
    struct ggml_tensor * mel_tensor = ggml_graph_get_tensor(gf_conv, "mel");
    if (!mel_tensor) {
        error_msg_ = "Failed to find mel tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    std::vector<float> transposed_mel(n_mel * n_frames);
    for (int m = 0; m < n_mel; ++m) {
        for (int f = 0; f < n_frames; ++f) {
            transposed_mel[f + m * n_frames] = mel_data[m * n_frames + f];
        }
    }
    
    ggml_backend_tensor_set(mel_tensor, transposed_mel.data(), 0, n_mel * n_frames * sizeof(float));
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf_conv) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute conv graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    struct ggml_tensor * embd_conv = ggml_graph_get_tensor(gf_conv, "embd_conv");
    if (!embd_conv) {
        error_msg_ = "Failed to find embd_conv tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    int64_t out_ctx = embd_conv->ne[1];
    int64_t out_state = embd_conv->ne[0];
    
    std::vector<float> conv_output(out_ctx * out_state);
    ggml_backend_tensor_get(embd_conv, conv_output.data(), 0, out_ctx * out_state * sizeof(float));
    
    ggml_backend_sched_reset(state_.sched);
    
    add_positional_embedding(model_, conv_output, out_ctx, n_state);
    
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_ASR_MAX_NODES + ggml_graph_overhead());
    
    struct ggml_init_params enc_params = {
        state_.compute_meta.size(),
        state_.compute_meta.data(),
        true,
    };
    
    struct ggml_context * enc_ctx = ggml_init(enc_params);
    struct ggml_cgraph * gf_enc = ggml_new_graph_custom(enc_ctx, QWEN3_ASR_MAX_NODES, false);
    
    const auto & hp = model_.hparams;
    const int n_head = hp.n_attention_heads;
    const int n_layer = hp.n_encoder_layers;
    const int n_state_head = n_state / n_head;
    const float eps = hp.layer_norm_eps;
    const float KQscale = 1.0f / sqrtf(float(n_state_head));
    
    struct ggml_tensor * inpL = ggml_new_tensor_2d(enc_ctx, GGML_TYPE_F32, n_state, out_ctx);
    ggml_set_name(inpL, "enc_input");
    ggml_set_input(inpL);
    
    struct ggml_tensor * cur = inpL;
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        {
            cur = ggml_norm(enc_ctx, inpL, eps);
            if (layer.attn_norm_w) {
                cur = ggml_mul(enc_ctx, cur, layer.attn_norm_w);
            }
            if (layer.attn_norm_b) {
                cur = ggml_add(enc_ctx, cur, layer.attn_norm_b);
            }
        }
        
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(enc_ctx, layer.attn_q_w, cur);
            if (layer.attn_q_b) {
                Qcur = ggml_add(enc_ctx, Qcur, layer.attn_q_b);
            }
            
            struct ggml_tensor * Kcur = ggml_mul_mat(enc_ctx, layer.attn_k_w, cur);
            if (layer.attn_k_b) {
                Kcur = ggml_add(enc_ctx, Kcur, layer.attn_k_b);
            }
            
            struct ggml_tensor * Vcur = ggml_mul_mat(enc_ctx, layer.attn_v_w, cur);
            if (layer.attn_v_b) {
                Vcur = ggml_add(enc_ctx, Vcur, layer.attn_v_b);
            }
            
            struct ggml_tensor * Q = ggml_permute(enc_ctx,
                ggml_reshape_3d(enc_ctx, Qcur, n_state_head, n_head, out_ctx),
                0, 2, 1, 3);
            
            struct ggml_tensor * K = ggml_permute(enc_ctx,
                ggml_reshape_3d(enc_ctx, Kcur, n_state_head, n_head, out_ctx),
                0, 2, 1, 3);
            
            struct ggml_tensor * KQ = ggml_mul_mat(enc_ctx, K, Q);
            
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_ext(enc_ctx, KQ, nullptr, KQscale, 0.0f);
            
            struct ggml_tensor * V = ggml_cont(enc_ctx, ggml_permute(enc_ctx,
                ggml_reshape_3d(enc_ctx, Vcur, n_state_head, n_head, out_ctx),
                1, 2, 0, 3));
            
            struct ggml_tensor * KQV = ggml_mul_mat(enc_ctx, V, KQ_soft_max);
            
            struct ggml_tensor * KQV_merged = ggml_permute(enc_ctx, KQV, 0, 2, 1, 3);
            
            cur = ggml_cont_2d(enc_ctx, KQV_merged, n_state, out_ctx);
        }
        
        {
            cur = ggml_mul_mat(enc_ctx, layer.attn_out_w, cur);
            if (layer.attn_out_b) {
                cur = ggml_add(enc_ctx, cur, layer.attn_out_b);
            }
        }
        
        cur = ggml_add(enc_ctx, cur, inpL);
        
        struct ggml_tensor * inpFF = cur;
        
        {
            {
                cur = ggml_norm(enc_ctx, inpFF, eps);
                if (layer.ffn_norm_w) {
                    cur = ggml_mul(enc_ctx, cur, layer.ffn_norm_w);
                }
                if (layer.ffn_norm_b) {
                    cur = ggml_add(enc_ctx, cur, layer.ffn_norm_b);
                }
            }
            
            cur = ggml_mul_mat(enc_ctx, layer.ffn_up_w, cur);
            if (layer.ffn_up_b) {
                cur = ggml_add(enc_ctx, cur, layer.ffn_up_b);
            }
            
            cur = ggml_gelu(enc_ctx, cur);
            
            cur = ggml_mul_mat(enc_ctx, layer.ffn_down_w, cur);
            if (layer.ffn_down_b) {
                cur = ggml_add(enc_ctx, cur, layer.ffn_down_b);
            }
        }
        
        inpL = ggml_add(enc_ctx, cur, inpFF);
    }
    
    cur = inpL;
    
    if (model_.ln_post_w) {
        cur = ggml_norm(enc_ctx, cur, eps);
        cur = ggml_mul(enc_ctx, cur, model_.ln_post_w);
        if (model_.ln_post_b) {
            cur = ggml_add(enc_ctx, cur, model_.ln_post_b);
        }
    }
    
    if (model_.proj1_w) {
        cur = ggml_mul_mat(enc_ctx, model_.proj1_w, cur);
        if (model_.proj1_b) {
            cur = ggml_add(enc_ctx, cur, model_.proj1_b);
        }
        cur = ggml_gelu(enc_ctx, cur);
    }
    
    if (model_.proj2_w) {
        cur = ggml_mul_mat(enc_ctx, model_.proj2_w, cur);
        if (model_.proj2_b) {
            cur = ggml_add(enc_ctx, cur, model_.proj2_b);
        }
    }
    
    ggml_set_name(cur, "embd_enc");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf_enc, cur);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf_enc)) {
        error_msg_ = "Failed to allocate encoder graph";
        ggml_free(enc_ctx);
        return false;
    }
    
    struct ggml_tensor * enc_input = ggml_graph_get_tensor(gf_enc, "enc_input");
    if (!enc_input) {
        error_msg_ = "Failed to find enc_input tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(enc_ctx);
        return false;
    }
    
    ggml_backend_tensor_set(enc_input, conv_output.data(), 0, out_ctx * n_state * sizeof(float));
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf_enc) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute encoder graph";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(enc_ctx);
        return false;
    }
    
    struct ggml_tensor * embd_enc = ggml_graph_get_tensor(gf_enc, "embd_enc");
    if (!embd_enc) {
        error_msg_ = "Failed to find embd_enc tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(enc_ctx);
        return false;
    }
    
    int64_t final_ctx = embd_enc->ne[1];
    int64_t final_state = embd_enc->ne[0];
    
    output.resize(final_ctx * final_state);
    ggml_backend_tensor_get(embd_enc, output.data(), 0, final_ctx * final_state * sizeof(float));
    
    ggml_backend_sched_reset(state_.sched);
    ggml_free(enc_ctx);
    
    return true;
}

bool AudioEncoder::encode_conv_only(const float * mel_data, int n_mel, int n_frames,
                                     std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    if (n_mel != model_.hparams.n_mel_bins) {
        error_msg_ = "Mel bins mismatch";
        return false;
    }
    
    const int n_state = model_.hparams.d_model;
    
    struct ggml_cgraph * gf_conv = build_graph_conv(n_frames);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf_conv)) {
        error_msg_ = "Failed to allocate conv graph";
        return false;
    }
    
    struct ggml_tensor * mel_tensor = ggml_graph_get_tensor(gf_conv, "mel");
    if (!mel_tensor) {
        error_msg_ = "Failed to find mel tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    std::vector<float> transposed_mel(n_mel * n_frames);
    for (int m = 0; m < n_mel; ++m) {
        for (int f = 0; f < n_frames; ++f) {
            transposed_mel[f + m * n_frames] = mel_data[m * n_frames + f];
        }
    }
    
    ggml_backend_tensor_set(mel_tensor, transposed_mel.data(), 0, n_mel * n_frames * sizeof(float));
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf_conv) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute conv graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    struct ggml_tensor * embd_conv = ggml_graph_get_tensor(gf_conv, "embd_conv");
    if (!embd_conv) {
        error_msg_ = "Failed to find embd_conv tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    int64_t out_ctx = embd_conv->ne[1];
    int64_t out_state = embd_conv->ne[0];
    
    output.resize(out_ctx * out_state);
    ggml_backend_tensor_get(embd_conv, output.data(), 0, out_ctx * out_state * sizeof(float));
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

}

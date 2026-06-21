#include "qwen3_asr.h"
#include "timing.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <cctype>
#include <map>

namespace qwen3_asr {

static constexpr float QWEN3_ASR_DEFAULT_CHUNK_SECONDS = 30.0f;

static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string trim_ascii(const std::string & s) {
    size_t begin = 0;
    while (begin < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[begin]);
        if (c >= 0x80 || !std::isspace(c)) break;
        ++begin;
    }

    size_t end = s.size();
    while (end > begin) {
        unsigned char c = static_cast<unsigned char>(s[end - 1]);
        if (c >= 0x80 || !std::isspace(c)) break;
        --end;
    }

    return s.substr(begin, end - begin);
}

static std::string strip_language_prefix(const std::string & text) {
    const std::string prefix = "language ";
    if (text.size() < prefix.size() || text.compare(0, prefix.size(), prefix) != 0) {
        return trim_ascii(text);
    }

    size_t pos = prefix.size();
    while (pos < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[pos]);
        if (c >= 0x80 || std::isspace(c) || c == '<') {
            break;
        }
        ++pos;
    }

    while (pos < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[pos]);
        if (c >= 0x80 || !std::isspace(c)) {
            break;
        }
        ++pos;
    }

    const std::string marker = "<asr_text>";
    if (pos + marker.size() <= text.size() && text.compare(pos, marker.size(), marker) == 0) {
        pos += marker.size();
    }

    return trim_ascii(text.substr(pos));
}

static std::string normalize_language_alias(const std::string & language) {
    std::string lang;
    lang.reserve(language.size());
    for (char c : language) {
        if (c == '-' || c == '_') {
            lang.push_back(' ');
        } else {
            lang.push_back((char)std::tolower((unsigned char)c));
        }
    }

    static const std::map<std::string, std::string> aliases = {
        {"zh", "Chinese"},
        {"zho", "Chinese"},
        {"chinese", "Chinese"},
        {"cn", "Chinese"},
        {"en", "English"},
        {"eng", "English"},
        {"english", "English"},
        {"ko", "Korean"},
        {"kor", "Korean"},
        {"korean", "Korean"},
        {"ja", "Japanese"},
        {"jpn", "Japanese"},
        {"japanese", "Japanese"},
    };

    auto it = aliases.find(lang);
    if (it != aliases.end()) {
        return it->second;
    }
    return language;
}

static bool language_token_id(const std::string & language, int32_t & token_id) {
    static const std::map<std::string, int32_t> token_ids = {
        {"Chinese", 8453},
        {"English", 6364},
        {"Korean", 27470},
        {"Japanese", 22695},
    };

    auto it = token_ids.find(normalize_language_alias(language));
    if (it == token_ids.end()) {
        return false;
    }
    token_id = it->second;
    return true;
}

Qwen3ASR::Qwen3ASR() = default;
Qwen3ASR::~Qwen3ASR() = default;

bool Qwen3ASR::load_model(const std::string & model_path) {
    int64_t t_start = get_time_ms();
    
    if (!encoder_.load_model(model_path)) {
        error_msg_ = "Failed to load audio encoder: " + encoder_.get_error();
        return false;
    }
    
    if (!decoder_.load_model(model_path)) {
        error_msg_ = "Failed to load text decoder: " + decoder_.get_error();
        return false;
    }
    
    generate_mel_filters(mel_filters_, QWEN_N_MELS, QWEN_N_FFT, QWEN_SAMPLE_RATE);
    
    model_loaded_ = true;
    
    int64_t t_end = get_time_ms();
    fprintf(stderr, "Model loaded in %lld ms\n", (long long)(t_end - t_start));
    
    return true;
}

transcribe_result Qwen3ASR::transcribe(const std::string & audio_path,
                                        const transcribe_params & params) {
    transcribe_result result;
    
    if (!model_loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }
    
    std::vector<float> samples;
    int sample_rate;
    
    if (!load_wav(audio_path, samples, sample_rate)) {
        result.error_msg = "Failed to load audio file: " + audio_path;
        return result;
    }
    
    if (sample_rate != QWEN_SAMPLE_RATE) {
        result.error_msg = "Audio must be 16kHz, got " + std::to_string(sample_rate) + " Hz";
        return result;
    }
    
    return transcribe_internal(samples.data(), samples.size(), params);
}

transcribe_result Qwen3ASR::transcribe(const float * samples, int n_samples,
                                        const transcribe_params & params) {
    transcribe_result result;
    
    if (!model_loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }
    
    return transcribe_internal(samples, n_samples, params);
}

transcribe_result Qwen3ASR::transcribe_internal(const float * samples, int n_samples,
                                                 const transcribe_params & params) {
    const float chunk_seconds = params.chunk_seconds > 0.0f ?
        params.chunk_seconds : QWEN3_ASR_DEFAULT_CHUNK_SECONDS;
    const int max_chunk_samples = params.chunk_seconds > 0.0f ?
        std::max(1, (int)std::llround(chunk_seconds * QWEN_SAMPLE_RATE)) : n_samples;

    if (params.chunk_seconds > 0.0f && n_samples > max_chunk_samples) {
        return transcribe_chunked(samples, n_samples, params);
    }

    return transcribe_single(samples, n_samples, params);
}

transcribe_result Qwen3ASR::transcribe_chunked(const float * samples, int n_samples,
                                               const transcribe_params & params) {
    transcribe_result result;
    const int64_t t_total_start = get_time_ms();
    const int chunk_samples = std::max(1, (int)std::llround(params.chunk_seconds * QWEN_SAMPLE_RATE));
    const int n_chunks = (n_samples + chunk_samples - 1) / chunk_samples;

    if (params.print_progress) {
        fprintf(stderr, "Long audio: %.2f s, splitting into %d chunks of %.2f s\n",
                (double)n_samples / QWEN_SAMPLE_RATE, n_chunks, (double)chunk_samples / QWEN_SAMPLE_RATE);
    }

    std::string combined_text;
    transcribe_params chunk_params = params;
    chunk_params.print_timing = false;

    for (int i = 0; i < n_chunks; ++i) {
        const int start = i * chunk_samples;
        const int end = std::min(start + chunk_samples, n_samples);
        const int len = end - start;

        if (params.print_progress) {
            fprintf(stderr, "Transcribing chunk %d/%d [%.2f, %.2f] s\n",
                    i + 1, n_chunks,
                    (double)start / QWEN_SAMPLE_RATE,
                    (double)end / QWEN_SAMPLE_RATE);
        }

        transcribe_result chunk = transcribe_single(samples + start, len, chunk_params);
        result.t_mel_ms += chunk.t_mel_ms;
        result.t_encode_ms += chunk.t_encode_ms;
        result.t_decode_ms += chunk.t_decode_ms;

        if (!chunk.success) {
            result.error_msg = "Chunk " + std::to_string(i + 1) + "/" +
                               std::to_string(n_chunks) + " failed: " + chunk.error_msg;
            result.t_total_ms = get_time_ms() - t_total_start;
            return result;
        }

        result.tokens.insert(result.tokens.end(), chunk.tokens.begin(), chunk.tokens.end());

        std::string chunk_text = strip_language_prefix(chunk.text);
        if (!chunk_text.empty()) {
            if (!combined_text.empty()) {
                combined_text.push_back('\n');
            }
            combined_text += chunk_text;
        }
    }

    result.text = combined_text;
    result.success = true;
    result.t_total_ms = get_time_ms() - t_total_start;

    if (params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Chunks:          %d\n", n_chunks);
        fprintf(stderr, "  Mel spectrogram: %lld ms\n", (long long)result.t_mel_ms);
        fprintf(stderr, "  Audio encoding:  %lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Text decoding:   %lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:           %lld ms\n", (long long)result.t_total_ms);
        fprintf(stderr, "  Tokens generated: %zu\n", result.tokens.size());
    }

    return result;
}

transcribe_result Qwen3ASR::transcribe_single(const float * samples, int n_samples,
                                              const transcribe_params & params) {
    transcribe_result result;
    int64_t t_total_start = get_time_ms();
    
    int64_t t_mel_start = get_time_ms();
    MelSpectrogram mel;
    {
        QWEN3_TIMER("mel_spectrogram");
        if (!log_mel_spectrogram(samples, n_samples, mel_filters_, mel, params.n_threads)) {
            result.error_msg = "Failed to compute mel spectrogram";
            return result;
        }
    }
    result.t_mel_ms = get_time_ms() - t_mel_start;
    
    if (params.print_progress) {
        fprintf(stderr, "Mel spectrogram: [%d, %d]\n", mel.n_mel, mel.n_len);
    }
    
    int64_t t_encode_start = get_time_ms();
    std::vector<float> audio_features;
    {
        QWEN3_TIMER("audio_encoding");
        if (!encoder_.encode(mel.data.data(), mel.n_mel, mel.n_len, audio_features)) {
            result.error_msg = "Failed to encode audio: " + encoder_.get_error();
            return result;
        }
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;
    
    const auto & text_hparams = encoder_.get_text_hparams();
    int32_t n_audio_frames = audio_features.size() / text_hparams.hidden_size;
    
    if (params.print_progress) {
        fprintf(stderr, "Audio features: [%d, %d]\n", n_audio_frames, text_hparams.hidden_size);
    }
    
    std::vector<int32_t> input_tokens = build_input_tokens(n_audio_frames, params.language);
    
    if (params.print_progress) {
        fprintf(stderr, "Input tokens: %zu\n", input_tokens.size());
    }
    
    int64_t t_decode_start = get_time_ms();
    std::vector<int32_t> output_tokens;
    if (!decode_greedy(input_tokens, audio_features, n_audio_frames, params, output_tokens)) {
        result.error_msg = "Decoding failed: " + error_msg_;
        return result;
    }
    result.t_decode_ms = get_time_ms() - t_decode_start;
    
    result.tokens = output_tokens;
    result.text = decoder_.decode_tokens(output_tokens);
    result.success = true;
    
    result.t_total_ms = get_time_ms() - t_total_start;
    
    if (params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Mel spectrogram: %lld ms\n", (long long)result.t_mel_ms);
        fprintf(stderr, "  Audio encoding:  %lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Text decoding:   %lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:           %lld ms\n", (long long)result.t_total_ms);
        fprintf(stderr, "  Tokens generated: %zu\n", output_tokens.size());
    }
    
    return result;
}

std::vector<int32_t> Qwen3ASR::build_input_tokens(int32_t n_audio_frames,
                                                   const std::string & language) {
    const auto & cfg = decoder_.get_config();
    
    std::vector<int32_t> tokens;
    tokens.reserve(n_audio_frames + 20);
    
    // Chat template format:
    // <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>...<|audio_end|><|im_end|>\n<|im_start|>assistant\n
    
    // Token IDs from Qwen3 tokenizer:
    // <|im_start|> = 151644
    // <|im_end|> = 151645
    // system = 8948
    // user = 872
    // assistant = 77091
    // \n = 198
    
    const int32_t im_start = 151644;
    const int32_t im_end = 151645;
    const int32_t system_token = 8948;
    const int32_t user_token = 872;
    const int32_t assistant_token = 77091;
    const int32_t newline = 198;
    
    // <|im_start|>system\n<|im_end|>\n
    tokens.push_back(im_start);
    tokens.push_back(system_token);
    tokens.push_back(newline);
    tokens.push_back(im_end);
    tokens.push_back(newline);
    
    // <|im_start|>user\n
    tokens.push_back(im_start);
    tokens.push_back(user_token);
    tokens.push_back(newline);
    
    // <|audio_start|><|audio_pad|>...<|audio_end|>
    tokens.push_back(cfg.audio_start_token_id);
    for (int32_t i = 0; i < n_audio_frames; ++i) {
        tokens.push_back(cfg.audio_pad_token_id);
    }
    tokens.push_back(cfg.audio_end_token_id);
    
    // <|im_end|>\n<|im_start|>assistant\n
    tokens.push_back(im_end);
    tokens.push_back(newline);
    tokens.push_back(im_start);
    tokens.push_back(assistant_token);
    tokens.push_back(newline);
    
    if (!language.empty()) {
        int32_t lang_token = 0;
        if (language_token_id(language, lang_token)) {
            const int32_t language_token = 11528;
            const int32_t asr_text = 151704;
            tokens.push_back(language_token);
            tokens.push_back(lang_token);
            tokens.push_back(asr_text);
        }
    }
    
    return tokens;
}

bool Qwen3ASR::decode_greedy(const std::vector<int32_t> & input_tokens,
                              const std::vector<float> & audio_features,
                              int32_t n_audio_frames,
                              const transcribe_params & params,
                              std::vector<int32_t> & output_tokens) {
    const auto & cfg = decoder_.get_config();
    
    int32_t n_ctx_needed = input_tokens.size() + params.max_tokens;
    if (!decoder_.init_kv_cache(n_ctx_needed)) {
        error_msg_ = "Failed to initialize KV cache: " + decoder_.get_error();
        return false;
    }
    
    std::vector<float> logits;
    
    // Audio pad tokens start after: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
    // That's 8 tokens before the first audio_pad
    int32_t audio_start_pos = 9;
    
    {
        QWEN3_TIMER("decode.initial_forward");
        if (!decoder_.forward_with_audio(
                input_tokens.data(), input_tokens.size(),
                audio_features.data(), n_audio_frames,
                audio_start_pos, 0, logits)) {
            error_msg_ = "Initial forward pass failed: " + decoder_.get_error();
            return false;
        }
    }
    
    int32_t vocab_size = cfg.vocab_size;
    int32_t n_input = input_tokens.size();
    
    int32_t next_token = sample_greedy(logits.data(), vocab_size);
    
    output_tokens.clear();
    output_tokens.push_back(next_token);
    
    if (progress_callback_) {
        progress_callback_(1, params.max_tokens);
    }
    
    int32_t n_past = n_input;
    
    while (next_token != cfg.eos_token_id && 
           (int32_t)output_tokens.size() < params.max_tokens) {
        
        std::vector<int32_t> single_token = {next_token};
        
        {
            QWEN3_TIMER("decode.token");
            if (!decoder_.forward(single_token.data(), 1, n_past, logits)) {
                error_msg_ = "Forward pass failed at token " + 
                             std::to_string(output_tokens.size()) + ": " + decoder_.get_error();
                return false;
            }
        }
        
        next_token = sample_greedy(logits.data(), vocab_size);
        output_tokens.push_back(next_token);
        
        n_past += 1;
        
        if (progress_callback_) {
            progress_callback_(output_tokens.size(), params.max_tokens);
        }
        
        if (params.print_progress && output_tokens.size() % 10 == 0) {
            fprintf(stderr, "Generated %zu tokens...\n", output_tokens.size());
        }
    }
    
    if (output_tokens.back() == cfg.eos_token_id) {
        output_tokens.pop_back();
    }
    
    return true;
}

int32_t Qwen3ASR::sample_greedy(const float * logits, int32_t vocab_size) {
    int32_t max_idx = 0;
    float max_val = logits[0];
    
    for (int32_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

void Qwen3ASR::set_progress_callback(progress_callback_t callback) {
    progress_callback_ = std::move(callback);
}

bool load_audio_file(const std::string & path, std::vector<float> & samples, int & sample_rate) {
    return load_wav(path, samples, sample_rate);
}

} // namespace qwen3_asr

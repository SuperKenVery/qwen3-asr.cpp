#!/usr/bin/env python3
"""Merge a split Qwen3-ASR text GGUF and mmproj GGUF for this runtime.

The current C++ loader expects a single GGUF with qwen3-asr metadata keys and
audio tensors named audio.encoder.*. Recent llama.cpp-style Qwen3-ASR exports
ship a text model plus a separate mmproj file with qwen3vl/clip metadata and
a.* tensor names. This script rewrites those names into the local convention.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import gguf


SKIP_COPIED_KEYS = {
    "general.architecture",
}


def field_value(field: Any) -> Any:
    value = field.contents()
    if isinstance(value, list):
        return [field_value_item(v) for v in value]
    return field_value_item(value)


def field_value_item(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def copy_field(writer: gguf.GGUFWriter, key: str, field: Any) -> None:
    if key.startswith("GGUF.") or key in SKIP_COPIED_KEYS:
        return

    value = field_value(field)
    value_type = field.types[0]
    sub_type = field.types[1] if value_type == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(key, value, value_type, sub_type=sub_type)


def get_field(reader: gguf.GGUFReader, key: str, default: Any = None) -> Any:
    field = reader.fields.get(key)
    if field is None:
        return default
    return field_value(field)


def infer_conv_channels(mmproj: gguf.GGUFReader) -> int:
    for tensor in mmproj.tensors:
        if tensor.name == "a.conv2d.1.bias":
            return int(tensor.shape[-1])
    for tensor in mmproj.tensors:
        if tensor.name == "a.conv2d.1.weight":
            return int(tensor.shape[-1])
    return 480


def rename_mmproj_tensor(name: str) -> str | None:
    direct = {
        "a.position_embd.weight": "audio.encoder.pos_embd.weight",
        "a.conv2d.1.weight": "audio.encoder.conv1.weight",
        "a.conv2d.1.bias": "audio.encoder.conv1.bias",
        "a.conv2d.2.weight": "audio.encoder.conv2.weight",
        "a.conv2d.2.bias": "audio.encoder.conv2.bias",
        "a.conv2d.3.weight": "audio.encoder.conv3.weight",
        "a.conv2d.3.bias": "audio.encoder.conv3.bias",
        "a.conv_out.weight": "audio.encoder.conv_out.weight",
        "a.post_ln.weight": "audio.encoder.ln_post.weight",
        "a.post_ln.bias": "audio.encoder.ln_post.bias",
        "mm.a.mlp.1.weight": "audio.encoder.proj1.weight",
        "mm.a.mlp.1.bias": "audio.encoder.proj1.bias",
        "mm.a.mlp.2.weight": "audio.encoder.proj2.weight",
        "mm.a.mlp.2.bias": "audio.encoder.proj2.bias",
    }
    if name in direct:
        return direct[name]

    match = re.fullmatch(r"a\.blk\.(\d+)\.(.+)", name)
    if not match:
        return None

    layer, suffix = match.groups()
    suffix_map = {
        "ln1.weight": "attn_norm.weight",
        "ln1.bias": "attn_norm.bias",
        "ln2.weight": "ffn_norm.weight",
        "ln2.bias": "ffn_norm.bias",
        "attn_q.weight": "attn_q.weight",
        "attn_q.bias": "attn_q.bias",
        "attn_k.weight": "attn_k.weight",
        "attn_k.bias": "attn_k.bias",
        "attn_v.weight": "attn_v.weight",
        "attn_v.bias": "attn_v.bias",
        "attn_out.weight": "attn_out.weight",
        "attn_out.bias": "attn_out.bias",
        "ffn_up.weight": "ffn_up.weight",
        "ffn_up.bias": "ffn_up.bias",
        "ffn_down.weight": "ffn_down.weight",
        "ffn_down.bias": "ffn_down.bias",
    }
    mapped = suffix_map.get(suffix)
    if mapped is None:
        return None
    return f"audio.encoder.blk.{layer}.{mapped}"


def add_compat_metadata(writer: gguf.GGUFWriter, text: gguf.GGUFReader, mmproj: gguf.GGUFReader) -> None:
    text_arch = get_field(text, "general.architecture", "qwen3vl")
    audio_arch = "clip"

    vocab_size = len(get_field(text, "tokenizer.ggml.tokens", []))
    if vocab_size:
        writer.add_uint32("qwen3-asr.vocab_size", vocab_size)

    writer.add_uint32("qwen3-asr.block_count", get_field(text, f"{text_arch}.block_count", 28))
    writer.add_uint32("qwen3-asr.embedding_length", get_field(text, f"{text_arch}.embedding_length", 1024))
    writer.add_uint32("qwen3-asr.feed_forward_length", get_field(text, f"{text_arch}.feed_forward_length", 3072))
    writer.add_uint32("qwen3-asr.attention.head_count", get_field(text, f"{text_arch}.attention.head_count", 16))
    writer.add_uint32("qwen3-asr.attention.head_count_kv", get_field(text, f"{text_arch}.attention.head_count_kv", 8))
    writer.add_uint32("qwen3-asr.attention.key_length", get_field(text, f"{text_arch}.attention.key_length", 64))
    writer.add_uint32("qwen3-asr.attention.value_length", get_field(text, f"{text_arch}.attention.value_length", 64))
    writer.add_float32("qwen3-asr.attention.layer_norm_rms_epsilon", get_field(text, f"{text_arch}.attention.layer_norm_rms_epsilon", 1e-6))
    writer.add_float32("qwen3-asr.rope.freq_base", get_field(text, f"{text_arch}.rope.freq_base", 1000000.0))

    writer.add_uint32("qwen3-asr.audio.start_token_id", 151669)
    writer.add_uint32("qwen3-asr.audio.end_token_id", 151670)
    writer.add_uint32("qwen3-asr.audio.pad_token_id", 151676)

    writer.add_uint32("audio.encoder_layers", get_field(mmproj, f"{audio_arch}.audio.block_count", 18))
    writer.add_uint32("audio.d_model", get_field(mmproj, f"{audio_arch}.audio.embedding_length", 896))
    writer.add_uint32("audio.attention_heads", get_field(mmproj, f"{audio_arch}.audio.attention.head_count", 14))
    writer.add_uint32("audio.ffn_dim", get_field(mmproj, f"{audio_arch}.audio.feed_forward_length", 3584))
    writer.add_uint32("audio.conv_channels", infer_conv_channels(mmproj))
    writer.add_uint32("audio.conv_out_dim", get_field(mmproj, f"{audio_arch}.audio.embedding_length", 896))
    writer.add_uint32("audio.num_mel_bins", get_field(mmproj, f"{audio_arch}.audio.num_mel_bins", 128))
    writer.add_uint32("audio.n_window_infer", 800)
    writer.add_float32("audio.layer_norm_eps", get_field(mmproj, f"{audio_arch}.audio.attention.layer_norm_epsilon", 1e-5))

    writer.add_uint32("text.hidden_size", get_field(text, f"{text_arch}.embedding_length", 1024))
    writer.add_uint32("text.decoder_layers", get_field(text, f"{text_arch}.block_count", 28))
    writer.add_uint32("text.attention_heads", get_field(text, f"{text_arch}.attention.head_count", 16))
    writer.add_uint32("text.num_key_value_heads", get_field(text, f"{text_arch}.attention.head_count_kv", 8))
    writer.add_uint32("text.intermediate_size", get_field(text, f"{text_arch}.feed_forward_length", 3072))
    writer.add_float32("text.rms_norm_eps", get_field(text, f"{text_arch}.attention.layer_norm_rms_epsilon", 1e-6))


def add_tensor(writer: gguf.GGUFWriter, name: str, tensor: Any) -> None:
    writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type)


def merge(text_path: Path, mmproj_path: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    text = gguf.GGUFReader(text_path)
    mmproj = gguf.GGUFReader(mmproj_path)

    writer = gguf.GGUFWriter(output_path, arch="qwen3-asr")

    for key, field in text.fields.items():
        copy_field(writer, key, field)
    for key, field in mmproj.fields.items():
        if key not in writer.kv_data[0]:
            copy_field(writer, key, field)

    add_compat_metadata(writer, text, mmproj)

    tensor_count = 0
    for tensor in text.tensors:
        add_tensor(writer, tensor.name, tensor)
        tensor_count += 1

    mapped_count = 0
    skipped = []
    for tensor in mmproj.tensors:
        mapped = rename_mmproj_tensor(tensor.name)
        if mapped is None:
            skipped.append(tensor.name)
            continue
        add_tensor(writer, mapped, tensor)
        mapped_count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Wrote {output_path}")
    print(f"Copied {tensor_count} text tensors and {mapped_count} audio/projector tensors")
    if skipped:
        print(f"Skipped {len(skipped)} unmapped mmproj tensors:")
        for name in skipped:
            print(f"  {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, type=Path, help="Main text GGUF")
    parser.add_argument("--mmproj", required=True, type=Path, help="Audio mmproj GGUF")
    parser.add_argument("--output", required=True, type=Path, help="Merged output GGUF")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    merge(args.text, args.mmproj, args.output, args.overwrite)


if __name__ == "__main__":
    main()

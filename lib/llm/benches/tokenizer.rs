// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use dynamo_llm::backend::Decoder;
use dynamo_llm::protocols::common::StopConditions;
use dynamo_llm::tokenizers::DecodeStream;
use dynamo_llm::tokenizers::hf::HuggingFaceTokenizer;
use dynamo_llm::tokenizers::traits::{Encoder, Tokenizer};
use dynamo_llm::types::TokenIdType;

const TEST_TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
);

/// Input Sequence Length for tokenizer
const TARGET_ISL: usize = 8_000;

// A string of length exactly 128 bytes.
const INPUT_STR: &str = "The cat sat by the window, watching raindrops race down the glass. Far thunder rumbled. She purred softly, feeling safe at home.";

/// `cargo bench -- encode` to run it
pub fn encode(c: &mut Criterion) {
    let test_str: &str = &INPUT_STR.repeat(TARGET_ISL / INPUT_STR.len());

    let encoder = HuggingFaceTokenizer::from_file(TEST_TOKENIZER).unwrap();
    let mut group = c.benchmark_group("encode-group");
    group.throughput(Throughput::Bytes(test_str.len() as u64));
    group.bench_function("tokenizer_encode", |b| {
        b.iter(|| {
            let _ = encoder.encode(black_box(test_str)).unwrap();
        })
    });
    group.finish();
}

pub fn decode(c: &mut Criterion) {
    const TEST_TOKS: [TokenIdType; 34] = [
        450, 6635, 3290, 491, 278, 3474, 29892, 21217, 1153, 513, 307, 567, 8175, 1623, 278, 12917,
        29889, 8413, 266, 5062, 364, 25443, 29889, 2296, 3708, 1127, 4964, 368, 29892, 11223, 9109,
        472, 3271, 29889,
    ];

    let mut group = c.benchmark_group("decode-group");
    group.throughput(Throughput::Elements(TEST_TOKS.len() as u64));
    group.bench_function("tokenizer_decoder", |b| {
        b.iter_with_setup(
            || {
                let tokenizer: Arc<dyn Tokenizer> =
                    Arc::new(HuggingFaceTokenizer::from_file(TEST_TOKENIZER).unwrap());
                let ds = DecodeStream::new(tokenizer, &[], false);
                Decoder::new(ds, StopConditions::default())
            },
            |mut decoder| {
                for tok in black_box(TEST_TOKS) {
                    let _ = decoder.step(tok).unwrap();
                }
            },
        )
    });
    group.finish();
}

pub fn decode_big(c: &mut Criterion) {
    const NUM_TOKENS: usize = 2048;

    const BIG_TEST_TOKS: [TokenIdType; NUM_TOKENS] = [450; NUM_TOKENS];
    let mut group = c.benchmark_group("decode-big-group");
    group.throughput(Throughput::Elements(NUM_TOKENS as u64));
    group.bench_function("tokenizer_decoder_big", |b| {
        b.iter_with_setup(
            || {
                let tokenizer: Arc<dyn Tokenizer> =
                    Arc::new(HuggingFaceTokenizer::from_file(TEST_TOKENIZER).unwrap());
                let ds = DecodeStream::new(tokenizer, &[], false);
                Decoder::new(ds, StopConditions::default())
            },
            |mut decoder| {
                for tok in black_box(&BIG_TEST_TOKS) {
                    let _ = decoder.step(*tok).unwrap();
                }
            },
        )
    });
    group.finish();
}

/// Benchmark for batch encoding - measures throughput when encoding multiple prompts simultaneously.
/// This is critical for inference servers that batch requests together.
/// `cargo bench -- encode_batch` to run it
pub fn encode_batch(c: &mut Criterion) {
    // Create a diverse batch of prompts with varying lengths
    const BATCH_SIZES: [usize; 3] = [8, 32, 64];

    let encoder = HuggingFaceTokenizer::from_file(TEST_TOKENIZER).unwrap();

    for batch_size in BATCH_SIZES {
        // Create batch with varying prompt lengths for realistic workload
        let batch: Vec<&str> = (0..batch_size)
            .map(|i| {
                // Vary prompt length: short, medium, long based on position
                match i % 3 {
                    0 => "Hello, how are you?",
                    1 => INPUT_STR,
                    _ => "The quick brown fox jumps over the lazy dog. This is a medium length prompt for testing batch encoding performance.",
                }
            })
            .collect();

        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        let mut group = c.benchmark_group(format!("encode-batch-{}", batch_size));
        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_function(format!("tokenizer_encode_batch_{}", batch_size), |b| {
            b.iter(|| {
                let _ = encoder.encode_batch(black_box(&batch)).unwrap();
            })
        });
        group.finish();
    }
}

/// Benchmark for encoding edge cases - measures performance on challenging inputs
/// `cargo bench -- encode_edge_cases` to run it
pub fn encode_edge_cases(c: &mut Criterion) {
    let encoder = HuggingFaceTokenizer::from_file(TEST_TOKENIZER).unwrap();

    let mut group = c.benchmark_group("encode-edge-cases");

    // Empty string
    group.bench_function("encode_empty", |b| {
        b.iter(|| {
            let _ = encoder.encode(black_box("")).unwrap();
        })
    });

    // Unicode-heavy text (emojis, CJK characters)
    let unicode_text = "Hello 👋 世界！🎉 こんにちは 🌍 مرحبا";
    group.throughput(Throughput::Bytes(unicode_text.len() as u64));
    group.bench_function("encode_unicode_heavy", |b| {
        b.iter(|| {
            let _ = encoder.encode(black_box(unicode_text)).unwrap();
        })
    });

    // Whitespace variations
    let whitespace_text = "   word   another\t\tword\n\nmore words   ";
    group.bench_function("encode_whitespace_heavy", |b| {
        b.iter(|| {
            let _ = encoder.encode(black_box(whitespace_text)).unwrap();
        })
    });

    // Repeated characters (tests tokenizer merge behavior)
    let repeated_chars = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    group.bench_function("encode_repeated_chars", |b| {
        b.iter(|| {
            let _ = encoder.encode(black_box(repeated_chars)).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, encode, decode, decode_big, encode_batch, encode_edge_cases);
criterion_main!(benches);

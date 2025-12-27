// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tokenizer Tests
//!
//! This module contains tests for the Tokenizer.
//!
//! For each tokenizer we use in production, we should have either a url to or a local copy
//! of either the tokenizer.json or the .model file.
//!
//! For a small set of common prompts, we need to have a hashable representation of the the encoding
//! object. We will precompute the hashes for each of these prompts for each tokenizer and store them
//! in a hashmap. We will then use these hashes to test that the tokenizer is working correctly. This
//! will detect if upstream dependency changes result in different/new behavior.

use dynamo_llm::tokenizers::traits::{Decoder, Encoder};
use dynamo_llm::tokenizers::*;
use std::collections::HashMap;
use std::sync::Arc;

// Additional imports for edge case tests
use dynamo_llm::tokenizers::{DecodeStream, SequenceDecoderOutput, StopSequenceDecoder};

const TEST_PROMPTS: [&str; 4] = [
    "deep learning is",
    "Deep learning is",
    "has anyone seen nemo lately",
    "another prompt",
];

const LONG_TEST_PROMPTS: [(&str, &str); 6] = [
    ("Tell me about the following text.", "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."),
    ("Tell me about the following text.", "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."),
    ("Tell me about the following text.", "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt."),
    ("Tell me about the following text.", "Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem."),
    // Note(jthomson04): Ishan asked me to add this one.
    ("Tell me about the following text.", "In the ancient realm of Tennisia, the very magic of the land is drawn from the sport itself. Forehands light the skies, backhands carve the earth, and serves rumble like thunder across kingdoms. At the center of this balance lie four sacred Grand Slam relics: the Sapphire Trophy of Melbourne, the Emerald Chalice of Paris, the Ruby Crown of London, and the Diamond Orb of New York. Together, they keep the game's spirit alive.
    But the relics are scattered, guarded by champions of legendary skill. The first is the Fire King of Clay, ruler of the crimson courts, whose topspin arcs blaze high and heavy, scorching all who dare stand across from him. The second is the Tempest Trickster, master of the baseline fortress, whose footwork and precision can turn back any storm, and whose returns arrive as if pulled by invisible strings. The third is the Shadow-Dancer of the Highlands, a tactician who thrives in the long rallies of twilight, changing pace and spin until opponents lose their rhythm. The fourth and final guardian is a towering Diamond Titan, a net-charging colossus whose volleys shatter the air itself.
    Into this arena of gods steps the Silver-Wristed Knight — a player of impossible grace, whose game is an art form. His quest: to claim each relic not for glory, but to restore harmony to the rankings of the realm.
    He travels across the Kingdom of Clay, where the points stretch like marathons and the air tastes of iron; through the Grasslands of London, where the ball skids low and the margins are razor-thin; over the Hard Courts of the East, where rallies turn into duels of endurance; and finally to the Cathedral of Lights in New York, where night matches burn with fevered energy.
    Each battle is played under enchanted floodlights, the lines patrolled by spectral line judges whose calls are final. The crowd's roar swells with every break point, and the Silver-Wristed Knight's racket glows brightest when the match teeters at deuce. There are moments when doubt grips him — when his serve falters or his touch deserts him — but each challenge teaches a new stroke, culminating in the legendary Forehand of Dawn.
    When the last relic is claimed, he stands not as a conqueror but as a custodian of the game, knowing that rivalries forge the very magic he protects. The balance is restored — until the next season begins."),
    // Emoji stress test
    ("Tell me about the following text.", "😀😃😄😁😆🥹😅😂🤣🥲☺️😊😇🙂🙃😉🤩😎 🤪🥳🤓🙄🤪😵👻")
];

const TINYLLAMA_TOKENIZER_PATH: &str = "tests/data/sample-models/TinyLlama_v1.1/tokenizer.json";

const HF_TOKENIZERS_LOCAL: [&str; 1] = [TINYLLAMA_TOKENIZER_PATH];

const HASHES: [(&str, [u64; 4]); 1] = [(
    TINYLLAMA_TOKENIZER_PATH,
    [
        1209591529327510910,
        4181375434596349981,
        6245658446118930933,
        5097285695902185237,
    ],
)];

fn compute_hashes_for_tokenizer<E: Encoder>(tokenizer: &E, prompts: &[&str]) -> Vec<u64> {
    prompts
        .iter()
        .map(|&prompt| {
            tokenizer
                .encode(prompt)
                .expect("Failed to encode prompt")
                .get_hash()
            // Assuming `get_hash` returns a `u64`
        })
        .collect()
}

#[test]
fn compute_hashes_hf() {
    let hash_map: HashMap<&str, [u64; 4]> = HASHES.iter().cloned().collect();

    for &tokenizer_name in HF_TOKENIZERS_LOCAL.iter() {
        let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_name)
            .expect("Failed to load HuggingFace tokenizer");

        let prompt_hashes = compute_hashes_for_tokenizer(&tokenizer, &TEST_PROMPTS);

        println!(
            "HF Tokenizer: {:?} Hashes: {:?}",
            tokenizer_name, prompt_hashes
        );

        assert_eq!(prompt_hashes, hash_map[tokenizer_name]);
    }
}

#[test]
fn test_hf_lifecycle() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load remote HuggingFace tokenizer");

    let encoding = tokenizer
        .encode(TEST_PROMPTS[0])
        .expect("Failed to encode prompt");

    let decoded = tokenizer
        .decode(encoding.token_ids(), false)
        .expect("Failed to decode token_ids");

    assert_eq!(decoded, TEST_PROMPTS[0]);
}

#[test]
fn test_sequence() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load remote HuggingFace tokenizer");

    let shared_tokenizer = Arc::new(tokenizer);

    // let tokenizer = shared_tokenizer.read().unwrap();

    let encoding = shared_tokenizer
        .encode(TEST_PROMPTS[0])
        .expect("Failed to encode prompt");

    let mut sequence = Sequence::new(shared_tokenizer.clone().into());
    sequence
        .append_text(TEST_PROMPTS[0])
        .expect("Failed to append prompt");

    assert_eq!(sequence.len(), encoding.token_ids().len());

    let mut decoder = Sequence::new(shared_tokenizer.clone().into());

    let mut output = String::new();
    for token_id in encoding.token_ids() {
        let text = decoder
            .append_token_id(*token_id)
            .expect("Failed to decode token_id");
        output.push_str(text.as_str());
    }

    assert_eq!(decoder.len(), sequence.len());
    assert_eq!(decoder.token_ids(), sequence.token_ids());
    assert_eq!(output, TEST_PROMPTS[0]);

    let mut decoder = DecodeStream::new(shared_tokenizer.clone(), &[], false);
    let mut output = String::new();
    for token_id in encoding.token_ids() {
        let text = decoder.step(*token_id).expect("Failed to decode token_id");
        if let Some(text) = text {
            output.push_str(text.as_str());
        }
    }
    assert_eq!(output, TEST_PROMPTS[0]);
}

#[test]
fn test_long_sequence_incremental_decode_with_prefill() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load remote HuggingFace tokenizer");

    let shared_tokenizer = Arc::new(tokenizer);

    for (input_text, output_text) in LONG_TEST_PROMPTS.iter() {
        let input_encoding = shared_tokenizer
            .encode(input_text)
            .expect("Failed to encode prompt");

        let output_encoding = shared_tokenizer
            .encode(output_text)
            .expect("Failed to encode prompt");

        let mut decoder =
            DecodeStream::new(shared_tokenizer.clone(), input_encoding.token_ids(), false);

        let mut output = String::new();
        for token_id in output_encoding.token_ids() {
            let text = decoder.step(*token_id).expect("Failed to decode token_id");
            if let Some(text) = text {
                output.push_str(text.as_str());
            }
        }

        assert_eq!(output.trim(), output_text.to_string());
    }
}

#[test]
fn test_decode_with_skip_special_tokens() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load remote HuggingFace tokenizer");

    // Create a sequence with special tokens:
    // <s> (token_id: 1) + "Hello world" + </s> (token_id: 2)
    let text = "Hello world";
    let encoding = tokenizer.encode(text).expect("Failed to encode text");
    let mut token_ids = vec![1]; // <s>
    token_ids.extend(encoding.token_ids());
    token_ids.push(2); // </s>

    // Decode with skip_special_tokens = false (should keep special tokens)
    let decoded_with_special = tokenizer
        .decode(&token_ids, false)
        .expect("Failed to decode with skip_special_tokens=false");

    // Decode with skip_special_tokens = true (should remove special tokens)
    let decoded_without_special = tokenizer
        .decode(&token_ids, true)
        .expect("Failed to decode with skip_special_tokens=true");

    // Validate exact matches on the entire decoded strings
    assert_eq!(decoded_with_special, "<s> Hello world</s>");
    assert_eq!(decoded_without_special, "Hello world");
}

// ============================================================================
// Edge Case Tests for Tokenizers
// ============================================================================

/// Test encoding an empty string - should produce empty token sequence
#[test]
fn test_encode_empty_string() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load tokenizer");

    let encoding = tokenizer.encode("").expect("Failed to encode empty string");
    assert!(
        encoding.token_ids().is_empty(),
        "Empty string should produce empty token sequence"
    );
}

/// Test encoding whitespace-only strings
#[test]
fn test_encode_whitespace_only() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load tokenizer");

    // Single space
    let encoding = tokenizer.encode(" ").expect("Failed to encode single space");
    // Result depends on tokenizer, just verify it doesn't panic

    // Multiple spaces
    let encoding = tokenizer
        .encode("     ")
        .expect("Failed to encode multiple spaces");

    // Mixed whitespace
    let encoding = tokenizer
        .encode("  \t\n  ")
        .expect("Failed to encode mixed whitespace");

    // Decode should produce valid output
    let decoded = tokenizer
        .decode(encoding.token_ids(), false)
        .expect("Failed to decode whitespace tokens");
    assert!(!decoded.is_empty() || encoding.token_ids().is_empty());
}

/// Test encoding Unicode characters (emojis, CJK, Arabic, etc.)
#[test]
fn test_encode_unicode_characters() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load tokenizer");

    // Emoji test
    let emoji_text = "Hello 👋 World 🌍";
    let encoding = tokenizer
        .encode(emoji_text)
        .expect("Failed to encode emoji text");
    let decoded = tokenizer
        .decode(encoding.token_ids(), false)
        .expect("Failed to decode emoji tokens");
    // Decoded should preserve the semantic content
    assert!(decoded.contains("Hello") && decoded.contains("World"));

    // CJK characters
    let cjk_text = "你好世界";
    let encoding = tokenizer
        .encode(cjk_text)
        .expect("Failed to encode CJK text");
    assert!(
        !encoding.token_ids().is_empty(),
        "CJK text should produce tokens"
    );

    // Arabic text
    let arabic_text = "مرحبا بالعالم";
    let encoding = tokenizer
        .encode(arabic_text)
        .expect("Failed to encode Arabic text");
    assert!(
        !encoding.token_ids().is_empty(),
        "Arabic text should produce tokens"
    );
}

/// Test batch encoding produces consistent results with single encoding
#[test]
fn test_encode_batch_consistency() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load tokenizer");

    let prompts = ["Hello world", "Testing batch", "Consistency check"];

    // Batch encode
    let batch_encodings = tokenizer
        .encode_batch(&prompts)
        .expect("Failed to batch encode");

    // Single encode each and compare
    for (i, prompt) in prompts.iter().enumerate() {
        let single_encoding = tokenizer.encode(prompt).expect("Failed to single encode");
        assert_eq!(
            batch_encodings[i].token_ids(),
            single_encoding.token_ids(),
            "Batch encoding should match single encoding for prompt: {}",
            prompt
        );
    }
}

/// Test encoding very long strings doesn't cause issues
#[test]
fn test_encode_long_string() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load tokenizer");

    // Create a long string (~100KB)
    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(2000);

    let encoding = tokenizer
        .encode(&long_text)
        .expect("Failed to encode long string");
    assert!(
        !encoding.token_ids().is_empty(),
        "Long string should produce tokens"
    );
    assert!(
        encoding.token_ids().len() > 1000,
        "Long string should produce many tokens"
    );

    // Verify decode is roughly equivalent
    let decoded = tokenizer
        .decode(encoding.token_ids(), false)
        .expect("Failed to decode long string");
    assert!(
        decoded.len() > 50000,
        "Decoded long string should be substantial"
    );
}

/// Test encoding strings with special characters and punctuation
#[test]
fn test_encode_special_characters() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load tokenizer");

    let special_chars = r#"Special: !@#$%^&*()_+-=[]{}|;':",.<>?/\`~"#;
    let encoding = tokenizer
        .encode(special_chars)
        .expect("Failed to encode special characters");
    assert!(
        !encoding.token_ids().is_empty(),
        "Special characters should produce tokens"
    );

    // Code-like content
    let code_content = r#"fn main() { println!("Hello, {}!", name); }"#;
    let encoding = tokenizer
        .encode(code_content)
        .expect("Failed to encode code content");
    let decoded = tokenizer
        .decode(encoding.token_ids(), false)
        .expect("Failed to decode code content");
    assert!(
        decoded.contains("fn") && decoded.contains("main"),
        "Code content should be preserved"
    );
}

/// Test encoding strings with repeated characters
#[test]
fn test_encode_repeated_characters() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load tokenizer");

    // Repeated single character
    let repeated_a = "a".repeat(100);
    let encoding = tokenizer
        .encode(&repeated_a)
        .expect("Failed to encode repeated 'a'");
    assert!(
        !encoding.token_ids().is_empty(),
        "Repeated characters should produce tokens"
    );

    // Repeated word
    let repeated_word = "word ".repeat(50);
    let encoding = tokenizer
        .encode(&repeated_word)
        .expect("Failed to encode repeated word");
    let decoded = tokenizer
        .decode(encoding.token_ids(), false)
        .expect("Failed to decode repeated word");
    // Count occurrences of "word"
    let word_count = decoded.matches("word").count();
    assert!(
        word_count >= 40,
        "Most repeated words should be preserved: found {}",
        word_count
    );
}

/// Test DecodeStream with incremental token additions
#[test]
fn test_decode_stream_incremental() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load tokenizer");

    let shared_tokenizer = Arc::new(tokenizer);

    let text = "This is a test of incremental decoding";
    let encoding = shared_tokenizer
        .encode(text)
        .expect("Failed to encode text");

    let mut decode_stream = DecodeStream::new(shared_tokenizer.clone(), &[], false);
    let mut accumulated = String::new();

    for &token_id in encoding.token_ids() {
        if let Some(chunk) = decode_stream.step(token_id).expect("Failed to step") {
            accumulated.push_str(&chunk);
        }
    }

    assert_eq!(
        accumulated.trim(),
        text,
        "Incremental decode should match original"
    );
}

/// Test StopSequenceDecoder correctly handles stop tokens
#[test]
fn test_stop_sequence_decoder_hidden_stop() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load tokenizer");

    let shared_tokenizer = Arc::new(tokenizer);

    // Build decoder with hidden stop token (EOS = 2 for TinyLlama)
    let mut decoder = StopSequenceDecoder::builder(shared_tokenizer.clone().into())
        .add_stop_token_id_hidden(2) // </s> token
        .build()
        .expect("Failed to build decoder");

    // Feed some tokens then the stop token
    let _ = decoder.append_token_id(450); // "The"
    let _ = decoder.append_token_id(3290); // " cat"

    // Feed the stop token - should stop
    let result = decoder.append_token_id(2).expect("Failed to append stop token");

    match result {
        SequenceDecoderOutput::Stopped => {
            // Expected - stop token is hidden
        }
        other => panic!("Expected Stopped, got {:?}", other),
    }

    assert!(decoder.is_complete(), "Decoder should be marked complete");
}

/// Test StopSequenceDecoder with visible stop token
#[test]
fn test_stop_sequence_decoder_visible_stop() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load tokenizer");

    let shared_tokenizer = Arc::new(tokenizer);

    // Build decoder with visible stop token
    let mut decoder = StopSequenceDecoder::builder(shared_tokenizer.clone().into())
        .add_stop_token_id_visible(29889) // "." period token
        .build()
        .expect("Failed to build decoder");

    // Feed tokens that form a sentence
    let _ = decoder.append_token_id(450); // "The"
    let _ = decoder.append_token_id(6635); // " cat"

    // Feed the stop token - should stop with text
    let result = decoder
        .append_token_id(29889)
        .expect("Failed to append stop token");

    match result {
        SequenceDecoderOutput::StoppedWithText(text) => {
            assert!(
                text.contains("cat") || text.contains("."),
                "Should contain accumulated text"
            );
        }
        other => panic!("Expected StoppedWithText, got {:?}", other),
    }

    assert!(decoder.is_complete(), "Decoder should be marked complete");
}

/// Test encoding/decoding roundtrip for various text types
#[test]
fn test_roundtrip_consistency() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load tokenizer");

    let test_cases = vec![
        "Simple text",
        "Text with numbers 123 and symbols !@#",
        "MixedCaseText",
        "   Padded text   ",
        "Multi\nline\ntext",
    ];

    for original in test_cases {
        let encoding = tokenizer
            .encode(original)
            .expect(&format!("Failed to encode: {}", original));
        let decoded = tokenizer
            .decode(encoding.token_ids(), false)
            .expect(&format!("Failed to decode: {}", original));

        // Trimmed versions should be equal (whitespace handling may differ)
        assert_eq!(
            decoded.trim(),
            original.trim(),
            "Roundtrip failed for: {}",
            original
        );
    }
}

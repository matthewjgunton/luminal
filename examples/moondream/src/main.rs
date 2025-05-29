use std::{
    fs::File,
    io::{self, Read, Write},
    path::Path,
    time::Instant,
};

use clap::Parser;
use itertools::Itertools;
use tokenizers::Tokenizer;

mod image_model;
mod image_process;
mod loader;
mod model;
mod text_model;

use crate::model::KVCache;
use luminal::prelude::*;

pub const BOS_ID: f32 = 50256.0;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct CLIArgs {
    /// Number of tokens to generate
    #[clap(short = 't', long = "gen_tokens", default_value = "256")]
    gen_tokens: i32,

    /// Prompt for the model
    #[clap(short = 'p', long = "prompt", default_value = include_str!("../prompts/merge_sort.txt"))]
    prompt: String,
}

fn main() {
    print!("Defining graph");
    let mut cx = Graph::new(); // main graph for text
    let cli_args = CLIArgs::parse();
    let tokenizer = Tokenizer::from_file("setup/tokenizer.json").unwrap();

    let mut input = cx.named_tensor("Input", (1, 's'));

    // Initialize an empty key-value cache
    let mut cache_src: Vec<KVCache> = (0..text_model::TXT_N_LAYERS)
        .map(|_| {
            (
                cx.named_tensor(
                    "Key Cache",
                    (1, text_model::TXT_N_HEADS, 'p', text_model::TXT_HEAD_DIM),
                ),
                cx.named_tensor(
                    "Value Cache",
                    (1, text_model::TXT_N_HEADS, 'p', text_model::TXT_HEAD_DIM),
                ),
            )
        })
        .collect();
    cache_src.set_dyn(
        vec![],
        (1, text_model::TXT_N_HEADS, 0, text_model::TXT_N_HEADS), // start off empty here
    );

    //load the model and the image so they're ready to be added to the graph during .execute()
    let model = model::Moondream::new(&mut cx);
    let mut model_weights = params(&model);
    loader::load("setup/moondream2.safetensors", &model, &mut cx);
    let img = cx.named_tensor("Image", (10, 3, 378, 378));
    loader::load_image_binary_with_path(&img, &mut cx, "./prompts/image.bin");

    // (1) prefill with image
    let kv_cache: Vec<(GraphTensor, GraphTensor)> =
        model::_run_vision_encoder(&model, img, &mut cx, &cache_src);

    // (2) prefill with query
    let input_ids = tokenizer
        .encode(&cli_args.prompt as &str, false)
        .unwrap()
        .get_ids()
        .to_vec();
    println!("input: {:?} {:?}", input.dims(), &cli_args.prompt as &str);
    input.set_dyn(
        input_ids.iter().map(|i| *i as f32).collect::<Vec<_>>(),
        (1, input_ids.len()),
    );
    let max = 730 + 1;
    let logits = model.forward((input, &kv_cache, 730, max));
    println!("out of forward()");

    cx.compile(
        (
            GenericCompiler::default(),
            #[cfg(feature = "metal")]
            (
                luminal_metal::MetalCompilerPreBuffer::<f32>::default(),
                luminal_metal::BufferCompilers::default(),
            ),
            #[cfg(feature = "cuda")]
            (luminal_cuda::CudaCompiler::<f16>::default(),),
            #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
            luminal_cpu::CPUCompiler::default(),
        ),
        (&mut input, &mut model_weights),
    );
    println!("done compiling");

    println!("before execute");
    cx.execute();
    println!("out");
}

fn argmax(dist: &[f32]) -> u32 {
    dist.iter()
        .position_max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap() as u32
}

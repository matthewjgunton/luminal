use std::{
    fs::File,
    io::{self, Read, Write},
    path::Path,
    time::Instant,
};

use clap::Parser;
use colored::Colorize;
use itertools::Itertools;
use luminal_nn::Embedding;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

mod image_model;
mod image_process;
mod loader;
mod model;
mod text_model;

use crate::model::KVCache;
use luminal::prelude::*;

fn main() {
    print!("Defining graph");
    let mut cx = Graph::new();
    let tokenizer = Tokenizer::from_file("setup/tokenizer.json").unwrap();

    // let toks = cx.tensor((1, 9)).set(vec![1 as f32; 1 * 9]);
    let toks = cx
        .tensor((10, 3, 378, 378))
        .set(vec![1 as f32; 10 * 3 * 378 * 378]);

    // Initialize an empty key-value cache
    let mut cache: Vec<KVCache> = Vec::new();

    let mut keyCache = cx.named_tensor(
        "Key Cache",
        (1, text_model::TXT_N_HEADS, 'p', text_model::TXT_HEAD_DIM),
    );

    let mut valueCache = cx.named_tensor(
        "Value Cache",
        (1, text_model::TXT_N_HEADS, 'p', text_model::TXT_HEAD_DIM),
    );

    cache.push((keyCache, valueCache));

    cache.set_dyn(
        vec![],
        (1, text_model::TXT_N_HEADS, 0, text_model::TXT_N_HEADS),
    );
    // Perform a forward pass through the model

    let model = model::Moondream::new(&mut cx);
    loader::load("setup/moondream2.safetensors", &model, &mut cx);

    let img = cx.named_tensor("Image", (10, 3, 378, 378));
    loader::load_image_binary_with_path(&img, &mut cx, "./prompts/image.bin");

    let (mut out, _) = model.forward((img, toks, &cache, &mut cx));

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
        (&mut out,),
    );

    println!("25");
    cx.execute();
}

use std::{
    fs::File,
    io::{self, Read, Write},
    time::Instant,
};

use clap::Parser;
use colored::Colorize;
use itertools::Itertools;
use luminal_nn::Embedding;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

mod loader;
mod model;

use crate::model::KVCache;
use luminal::prelude::*;

fn main() {
    print!("Defining graph");
    let mut cx = Graph::new();

    // let toks = cx.tensor((1, 9)).set(vec![1 as f32; 1 * 9]);
    let toks = cx
        .tensor((10, 3, 378, 378))
        .set(vec![1 as f32; 10 * 3 * 378 * 378]);
    let mul = cx.tensor((1, 32, 9, 16)).set(vec![2 as f32; 1 * 4608]);
    let mu2 = cx.tensor((9, 16)).set(vec![2 as f32; 1 * 144]);
    let out = mul * mu2;
    println!("{:?} * {:?} = {:?}", mul.dims(), mu2.dims(), out.dims());

    // Initialize an empty key-value cache
    let mut cache: Vec<KVCache> = Vec::new();

    let mut keyCache = cx.named_tensor(
        "Key Cache",
        (1, model::TXT_N_HEADS, 'p', model::TXT_HEAD_DIM),
    );

    let mut valueCache = cx.named_tensor(
        "Value Cache",
        (1, model::TXT_N_HEADS, 'p', model::TXT_HEAD_DIM),
    );

    cache.push((keyCache, valueCache));

    cache.set_dyn(vec![], (1, model::TXT_N_HEADS, 0, model::TXT_N_HEADS));
    // Perform a forward pass through the model

    let model = model::Moondream::new(&mut cx);
    loader::load("setup/moondream2.safetensors", &model, &mut cx);

    println!("ME");
    let (mut out, _) = model.forward((toks, &cache));

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

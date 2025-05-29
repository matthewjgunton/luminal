use std::io::Read;
use std::path::Path;
use std::{fs::File, io::Seek};

use luminal::{op::Function, prelude::*};
use memmap2::{Mmap, MmapOptions};
use safetensors::{tensor, Dtype, SafeTensors};

pub fn load_image_binary_with_path(img: &GraphTensor, graph: &mut Graph, file_path: &str) {
    let path = file_path.to_string();

    if let Some(loading_node) = graph
        .graph
        .node_weight_mut(img.id)
        .and_then(|op| op.as_any_mut().downcast_mut::<Function>())
    {
        loading_node.1 = Box::new(move |_| {
            let mut file: File =
                File::open(&path).unwrap_or_else(|_| panic!("Failed to open file: {}", path));

            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)
                .unwrap_or_else(|_| panic!("Failed to read file: {}", path));

            let expected_size = 10 * 3 * 378 * 378 * 4;

            if buffer.len() != expected_size {
                panic!(
                    "File size mismatch. Expected {} bytes, got {} bytes",
                    expected_size,
                    buffer.len()
                );
            }

            let data: Vec<f32> = buffer
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            vec![Tensor::new(data)]
        });
    }
}

pub fn load<M: SerializeModule>(path: &str, model: &M, graph: &mut Graph) {
    for (weight_name, node_index) in param_dict(model) {
        if let Some(loading_node) = graph
            .graph
            .node_weight_mut(node_index)
            .and_then(|op| op.as_any_mut().downcast_mut::<Function>())
        {
            let path = path.to_string();
            loading_node.1 = Box::new(move |_| {
                let file = File::open(&path).unwrap();
                let mmap = unsafe { Mmap::map(&file).unwrap() };
                let safetensors = SafeTensors::deserialize(&mmap).unwrap();

                if let Ok(tensor_view) = safetensors.tensor(&weight_name.replace('/', ".")) {
                    let data: Vec<f32> = match tensor_view.dtype() {
                        Dtype::F32 => tensor_view
                            .data()
                            .chunks_exact(4)
                            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                        Dtype::F16 => tensor_view
                            .data()
                            .chunks_exact(2)
                            .map(|c| f16::from_ne_bytes([c[0], c[1]]).to_f32())
                            .collect(),
                        _ => panic!("{:?} is not a supported dtype", tensor_view.dtype()),
                    };
                    return vec![Tensor::new(data)];
                }

                panic!("Tensor \"{weight_name}\" not found in files");
            });
        }
    }
}

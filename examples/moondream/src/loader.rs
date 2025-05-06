use std::io::Read;
use std::path::Path;
use std::{fs::File, io::Seek};

use luminal::{op::Function, prelude::*};
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors};

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
                // println!("Looking for tensor name: {}", weight_name.replace('/', "."));
                // println!("All keys: {:?}", safetensors.names());

                if let Ok(tensor_view) = safetensors.tensor(&weight_name.replace('/', ".")) {
                    // println!("Shape of tensor: {:?}", tensor_view.shape());
                    // println!("Assigning tensor {weight_name} to node index {node_index:?}");
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
                    // if weight_name.replace('/', ".") == "model.vision.patch_emb.weight" {
                    //     // print first row (VIS_DIM entries) and first 10 values of that row
                    //     let row0 = &data[0..1152];
                    //     println!("[debug] first row.len={} → {:?}", row0.len(), &row0[..10]);
                    // }
                    return vec![Tensor::new(data)];
                }

                panic!("Tensor \"{weight_name}\" not found in files");
            });
        }
    }
}

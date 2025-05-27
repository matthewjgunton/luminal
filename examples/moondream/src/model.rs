use luminal::{hl_ops::binary::F32Pow, prelude::*};
use luminal_nn::*;

use crate::image_model::{self, *};
use crate::text_model::{self, *};

pub const BOS_ID: f32 = 50256.0;

pub struct Moondream {
    vision_encoder: VisionEncoder,
    text_encoder: Embedding,
    text_decoder: Vec<TextBlock>,
    vision_projector: VisionProjector,
    lm_head: LmHead,
}
impl Moondream {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            vision_encoder: VisionEncoder::new(cx),
            text_encoder: Embedding::new(TXT_VOCAB_SIZE, TXT_DIM, cx),
            text_decoder: (0..TXT_N_LAYERS).map(|_| TextBlock::new(cx)).collect(),
            vision_projector: VisionProjector::new(cx),
            lm_head: LmHead::new(cx),
        }
    }
}

pub type KVCache = (GraphTensor, GraphTensor);

// this may need to be prefill rather than forwards...
// most important part of this is to fill the KV cache with the images + pos_ids
impl Module<(GraphTensor, GraphTensor, &[KVCache], &mut Graph, bool)> for Moondream {
    // Args: (image_tensor, token_ids, kv_cache[])
    type Output = (GraphTensor, Vec<KVCache>);
    fn forward(
        &self,
        (img, toks, cache, cx, is_prefill): (
            GraphTensor,
            GraphTensor,
            &[KVCache],
            &mut Graph,
            bool,
        ),
    ) -> Self::Output {
        img.diff("./prompts/image.bin", ATOL, RTOL);

        //IMAGE MODEL calls
        //(1) prepare_crops >> this is input into the model
        //(2) _vis_enc >> this is done below
        let x = self.vision_encoder.forward(img);
        x.diff("./bins/vis_enc_outputs.bin", ATOL, RTOL);

        //(3) reconstruct
        let global_features = x
            .slice((..Expression::from(0), .., ..))
            .reshape((VIS_NUM_PATCHES, VIS_DIM)) // unsqueeze effectively
            .contiguous();
        let (reconstructed, (output_h, output_w, channels_usize)) =
            image_model::reconstruct_from_crops(x);

        let normalized = cx
            .tensor((output_h, output_w, channels_usize))
            .set(reconstructed);

        //(4)_vis_proj [pending]
        let img_emb = self.vision_projector.forward((global_features, normalized)); // (expecting (729, 2048)

        //(5) text encoding embeddings
        let bos_tensor = cx.tensor((1, 1)).set(vec![BOS_ID]); // (expecting 1,1,2048)
        self.text_encoder.forward(bos_tensor);
        let mut prompt = img_emb
            .reshape((1, VIS_NUM_PATCHES, TXT_DIM))
            .contiguous()
            .concat_along(bos_tensor, 1)
            .contiguous();

        let (b, t, c) = prompt.dims3();
        let mut pos_ids: Vec<usize> = (0..t.to_usize().unwrap()).collect(); // t should be 730 here

        //(6) text decoder
        for layer in 0..self.text_decoder.len() {
            let (y, cache) = self.text_decoder[layer].forward((prompt, cache[0], &pos_ids));
            prompt = y;
        }
        // ---- at this stage it is ready to begin the decoder loop

        // (7) lm_head:

        let new = Vec::with_capacity(TXT_N_LAYERS);

        (prompt, new)
    }
}

///////////////////
///////////////////
/// Serializations
///////////////////
/// ///////////////////

impl SerializeModule for Moondream {
    fn serialize(&self, s: &mut Serializer) {
        // Vision branch
        s.module("model/vision", &self.vision_encoder);
        // s.module("model/vision/proj_mlp", &self.vis_proj);
        // s.module("model/region", &self.region);

        // Text branch
        s.tensor("model/text/wte", self.text_encoder.weight);
        for (i, blk) in self.text_decoder.iter().enumerate() {
            s.module(&format!("model/text/blocks/{i}"), blk);
        }
        s.module("model/text", &self.lm_head);
        // s.module("model/text/post_ln", &self.txt_norm);
        // s.module("model/text/lm_head", &self.lm_head);
    }
}

//    weight_map = {
//         "vision_encoder.encoder.model.visual.patch_embed.linear.weight": vision[
//             "patch_emb"
//         ].weight,
//         "vision_encoder.encoder.model.visual.patch_embed.linear.bias": vision[
//             "patch_emb"
//         ].bias,
//         "vision_encoder.encoder.model.visual.pos_embed": vision.pos_emb,
//         "vision_encoder.encoder.model.visual.norm.weight": vision["post_ln"].weight,
//         "vision_encoder.encoder.model.visual.norm.bias": vision["post_ln"].bias,
//         "vision_encoder.projection.mlp.fc1.weight": vision["proj_mlp"]["fc1"].weight,
//         "vision_encoder.projection.mlp.fc1.bias": vision["proj_mlp"]["fc1"].bias,
//         "vision_encoder.projection.mlp.fc2.weight": vision["proj_mlp"]["fc2"].weight,
//         "vision_encoder.projection.mlp.fc2.bias": vision["proj_mlp"]["fc2"].bias,
//     }

// this will need to be connected, not matching

// for i in range(len(model.vision["blocks"])):
//     prefix = f"vision_encoder.encoder.model.visual.blocks.{i}"
//     blk = model.vision["blocks"][i]
//     weight_map.update(
//         {
//             f"{prefix}.norm1.weight": blk["ln1"].weight,
//             f"{prefix}.norm1.bias": blk["ln1"].bias,
//             f"{prefix}.norm2.weight": blk["ln2"].weight,
//             f"{prefix}.norm2.bias": blk["ln2"].bias,
//             f"{prefix}.attn.qkv.weight": blk["attn"]["qkv"].weight,
//             f"{prefix}.attn.qkv.bias": blk["attn"]["qkv"].bias,
//             f"{prefix}.attn.proj.weight": blk["attn"]["proj"].weight,
//             f"{prefix}.attn.proj.bias": blk["attn"]["proj"].bias,
//             f"{prefix}.mlp.fc1.weight": blk["mlp"]["fc1"].weight,
//             f"{prefix}.mlp.fc1.bias": blk["mlp"]["fc1"].bias,
//             f"{prefix}.mlp.fc2.weight": blk["mlp"]["fc2"].weight,
//             f"{prefix}.mlp.fc2.bias": blk["mlp"]["fc2"].bias,
//         }
//     )

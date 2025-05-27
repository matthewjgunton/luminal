use luminal::{hl_ops::binary::F32Pow, prelude::*};
use luminal_nn::*;

use crate::model::*;
use crate::text_model::*;

pub const VIS_DIM: usize = 1152;
pub const VIS_PATCH_SIZE: usize = 14;
pub const VIS_PATCH_DIM: usize = VIS_PATCH_SIZE * VIS_PATCH_SIZE * 3; // 588
pub const VIS_NUM_PATCHES: usize = 729;
pub const VIS_FF_DIM: usize = 4304;
pub const VIS_N_LAYERS: usize = 27; //27; keeping it truly simple for now
pub const VIS_ENC_N_HEADS: usize = 16;
pub const VIS_HEAD_DIM: usize = VIS_DIM / VIS_ENC_N_HEADS;

// guessing: (1) redo attention for encoding head as parameter

// ATOL + RTOL * b.abs() allows for a
pub const ATOL: f32 = 1e-4;
pub const RTOL: f32 = 5e-3;

pub struct VisionAttention {
    qkv: Linear,
    proj: Linear,
}
impl VisionAttention {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            qkv: Linear::new_permuted(VIS_DIM, VIS_DIM * 3, true, cx), // shape here is up for debate
            proj: Linear::new_permuted(VIS_DIM, VIS_DIM, true, cx), // shape here is up for debate
        }
    }
}
// like text but no mask & no rotary pos
impl Module<GraphTensor> for VisionAttention {
    type Output = GraphTensor; // (b, n_tokens, VIS_DIM)
    fn forward(&self, x: GraphTensor) -> Self::Output {
        let (bsz, q_len, d_model) = x.dims3();
        let head_dim = d_model / VIS_ENC_N_HEADS;

        let qkv = self.qkv.forward(x);
        let q = qkv
            .slice((.., .., ..Expression::from(VIS_DIM)))
            .reshape((bsz, q_len, VIS_ENC_N_HEADS, head_dim))
            .permute((0, 2, 1, 3))
            .contiguous();
        let k = qkv
            .slice((
                ..,
                ..,
                Expression::from(VIS_DIM)..Expression::from(2 * VIS_DIM),
            ))
            .reshape((bsz, q_len, VIS_ENC_N_HEADS, head_dim))
            .permute((0, 2, 1, 3))
            .contiguous();
        let v = qkv
            .slice((
                ..,
                ..,
                Expression::from(2 * VIS_DIM)..Expression::from(3 * VIS_DIM),
            ))
            .reshape((bsz, q_len, VIS_ENC_N_HEADS, head_dim))
            .permute((0, 2, 1, 3))
            .contiguous();

        let mut att = q.matmul(k.permute((0, 1, 3, 2))); // dims are right, this is likely correct

        let sqrt_dk = (VIS_HEAD_DIM as f32).sqrt();
        att = att * (1.0 / sqrt_dk);
        att = att.softmax(3);

        let out = att
            .matmul(v)
            .permute((0, 2, 1, 3))
            .reshape((bsz, q_len, d_model));

        (self.proj.forward(out))
    }
}

pub struct VisionBlock {
    ln1: LayerNorm,
    ln2: LayerNorm,
    attn: VisionAttention,
    mlp: Mlp,
}
impl VisionBlock {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            ln1: LayerNorm::new(VIS_DIM, true, true, true, 1e-5, cx),
            ln2: LayerNorm::new(VIS_DIM, true, true, true, 1e-5, cx),
            attn: VisionAttention::new(cx),
            mlp: Mlp::new_with_bias(VIS_DIM, VIS_FF_DIM, cx),
        }
    }
}

impl Module<GraphTensor> for VisionBlock {
    type Output = GraphTensor; // (b, n_tokens, VIS_DIM)
    fn forward(&self, mut x: GraphTensor) -> Self::Output {
        let ln_x1 = self.ln1.forward(x);
        // make a key value cache maybe?
        let l_attn = self.attn.forward(ln_x1);
        x = x + l_attn;

        // x.diff("./bins/block_mid.bin", DIFF_THRESHOLD);

        let ln_x2 = self.ln2.forward(x);
        x = x + self.mlp.forward(ln_x2);

        // x.diff("./bins/block_end.bin", DIFF_THRESHOLD); // broken
        x
    }
}

fn create_patches(mut x: GraphTensor) -> GraphTensor {
    let (b, c, h, w) = x.dims4();
    let p1 = VIS_PATCH_SIZE;

    // Step 1: Split H and W dimensions into patches
    //[B, C, H/P1, P1, W/P2, P2]
    x = x.reshape((b, c, h / p1, p1, w / p1, p1));
    // x = x.reshape(B, C, H // P1, P1, W // P2, P2)

    // # Step 2: Rearrange dimensions to match target shape
    // # [B, H/P1, W/P2, C, P1, P2]
    x = x.permute((0, 2, 4, 1, 3, 5));

    // # Step 3: Combine dimensions to get final shape
    // # [B, (H/P1)*(W/P2), C*P1*P2]
    x = x.reshape((b, (h / p1) * (w / p1), c * p1 * p1));

    x
}
pub struct VisionEncoder {
    linear: Linear,
    pos_emb: GraphTensor,
    blocks: Vec<VisionBlock>,
    ln: LayerNorm,
}
impl VisionEncoder {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            linear: Linear::new_permuted(VIS_PATCH_DIM, VIS_DIM, true, cx),
            pos_emb: cx
                .named_tensor("pos_emb", (VIS_NUM_PATCHES, VIS_DIM))
                .expand(0, 1),
            blocks: (0..VIS_N_LAYERS).map(|_| VisionBlock::new(cx)).collect(),
            ln: LayerNorm::new(VIS_DIM, true, true, true, 1e-5, cx),
        }
    }
}
impl Module<GraphTensor> for VisionEncoder {
    type Output = GraphTensor;
    fn forward(&self, img: GraphTensor) -> Self::Output {
        img.diff("./prompts/image.bin", ATOL, RTOL);
        let mut x = create_patches(img);
        x.diff("./bins/create_patches.bin", ATOL, RTOL);

        self.linear
            .weight
            .diff("./bins/linear_weights.bin", ATOL, RTOL);

        self.linear
            .bias
            .unwrap()
            .diff("./bins/linear_bias.bin", ATOL, RTOL);

        x = self.linear.forward(x);
        x.diff("./bins/linear.bin", ATOL, RTOL);

        // self.pos_emb.diff("./bins/wpos_emb.bin", DIFF_THRESHOLD);
        println!("{:?} + {:?}", self.pos_emb.dims(), x.dims());
        x = x + self.pos_emb;

        println!("= {:?}", x.dims());
        // x.diff("./bins/pos_emb.bin", DIFF_THRESHOLD);
        x.diff("./bins/vis_enc_linpos.bin", ATOL, RTOL);

        for layer in 0..self.blocks.len() {
            x = self.blocks[layer].forward(x);
        }
        x.diff("./bins/vis_attn.bin", ATOL, RTOL);

        x = self.ln.forward(x);
        x
    }
}

pub fn reconstruct_from_crops(crops: GraphTensor) -> (Vec<f32>, (usize, usize, usize)) {
    let tiling_h = 378;
    let tiling_w = 378;
    let (range, crop_height, crop_width, channels) = crops.dims4();
    let crops_data = crops.data();

    let channels_usize = channels.to_usize().unwrap();
    let range_usize = range.to_usize().unwrap();
    let crop_height_usize = crop_height.to_usize().unwrap();
    let crop_width_usize = crop_width.to_usize().unwrap();

    let margin_pixels = 4 * VIS_PATCH_SIZE; // 4 = overlap margin

    let output_h = (crop_height_usize * margin_pixels) * tiling_h + 2 * margin_pixels;
    let output_w = (crop_width_usize * margin_pixels) * tiling_w + 2 * margin_pixels;

    let mut reconstructed = vec![0.0; output_h * output_w * channels_usize];

    for i in 0..range_usize {
        let tile_y = i / tiling_w;
        let tile_x = i % tiling_w;

        // Determine which part of the crop to copy
        let x_start = if tile_x == 0 { 0 } else { margin_pixels };
        let x_end = if tile_x == tiling_w - 1 {
            crop_width_usize
        } else {
            (crop_width_usize - margin_pixels)
        };

        let y_start = if tile_y == 0 { 0 } else { margin_pixels };
        let y_end = if tile_y == tiling_h - 1 {
            crop_height_usize
        } else {
            (crop_height_usize - margin_pixels)
        };

        // Calculate where this piece belongs in the output
        let out_x = tile_x * (crop_width_usize - 2 * margin_pixels);
        let out_y = tile_y * (crop_height_usize - 2 * margin_pixels);
        // Copy the relevant region into the output
        for y in y_start..y_end {
            for x in x_start..x_end {
                for c in 0..channels.to_usize().unwrap() {
                    let in_idx = (y * crop_width_usize + x) * channels_usize + c;
                    let out_x_global = out_x + x - x_start;
                    let out_y_global = out_y + y - y_start;
                    let out_idx = (out_y_global * output_w + out_x_global) * channels_usize + c;
                    if out_idx < reconstructed.len() && in_idx < crops_data.len() {
                        reconstructed[out_idx] = crops_data[in_idx];
                    }
                }
            }
        }
    }
    (reconstructed, (output_h, output_w, channels_usize))
}

pub struct VisionProjector {
    mlp: Mlp,
    adaptive_avg_pool: AdaptiveAvgPool2D,
}

impl VisionProjector {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            adaptive_avg_pool: AdaptiveAvgPool2D::new((VIS_N_LAYERS, VIS_N_LAYERS)),
            mlp: Mlp::new_different_intermediary(VIS_DIM * 2, TXT_FF_DIM, TXT_DIM, cx),
        }
    }
}

impl Module<(GraphTensor, GraphTensor)> for VisionProjector {
    type Output = GraphTensor;
    fn forward(
        &self,
        (global_features, mut reconstructed): (GraphTensor, GraphTensor),
    ) -> Self::Output {
        reconstructed = reconstructed.permute((2, 0, 1));
        reconstructed = self.adaptive_avg_pool.forward(reconstructed);

        reconstructed = reconstructed
            .permute((1, 2, 0))
            .reshape((729, VIS_N_LAYERS))
            .contiguous();
        let (a, b) = reconstructed.dims2();

        let final_features = reconstructed
            .reshape((a, b, 1))
            .contiguous()
            .concat_along(global_features.reshape((a, b, 1)).contiguous(), 2);

        let mlp = self.mlp.forward(final_features);
        mlp
    }
}

// starting from /vision
impl SerializeModule for VisionEncoder {
    fn serialize(&self, s: &mut Serializer) {
        s.module("patch_emb", &self.linear);
        s.module("post_ln", &self.ln);
        s.tensor("pos_emb", self.pos_emb);
        for (i, blk) in self.blocks.iter().enumerate() {
            s.module(&format!("blocks/{i}"), blk);
        }
    }
}

impl SerializeModule for VisionBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("ln1", &self.ln1);
        s.module("ln2", &self.ln2);
        s.module("attn", &self.attn);
        s.module("mlp", &self.mlp);
    }
}

impl SerializeModule for VisionAttention {
    fn serialize(&self, s: &mut Serializer) {
        s.module("qkv", &self.qkv); // weight key: ".../qkv"
        s.module("proj", &self.proj); // weight key: ".../proj"
    }
}

use luminal::{hl_ops::binary::F32Pow, prelude::*};
use luminal_nn::*;

use crate::model::*;
use crate::text_model::*;

pub const VIS_DIM: usize = 1152;
pub const VIS_PATCH_SIZE: usize = 14;
pub const VIS_PATCH_DIM: usize = VIS_PATCH_SIZE * VIS_PATCH_SIZE * 3; // 588
pub const VIS_NUM_PATCHES: usize = 729;
pub const VIS_FF_DIM: usize = 4304;
pub const VIS_N_LAYERS: usize = 27;
pub const VIS_ENC_N_HEADS: usize = 16;
pub const VIS_HEAD_DIM: usize = VIS_DIM / VIS_ENC_N_HEADS;

pub const ATOL: f32 = 1e-4;
pub const RTOL: f32 = 5e-3;

pub struct VisionModel {
    vision_projector: VisionProjector,
    vision_encoder: VisionEncoder,
}
impl VisionModel {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            vision_projector: VisionProjector::new(cx),
            vision_encoder: VisionEncoder::new(cx),
        }
    }
}

impl Module<GraphTensor> for VisionModel {
    type Output = GraphTensor;
    fn forward(&self, img: GraphTensor) -> Self::Output {
        let x = self.vision_encoder.forward(img);

        let (a, _, _) = x.dims3();

        let global_features = x
            .slice((..Expression::from(0), .., ..))
            .reshape((VIS_NUM_PATCHES, VIS_DIM)) // unsqueeze effectively
            .contiguous();
        let local_features = x
            .slice((Expression::from(1).., .., ..))
            .reshape((a - 1, VIS_N_LAYERS, VIS_N_LAYERS, VIS_DIM)) // unsqueeze effectively
            .contiguous();
        let reconstructed = self::reconstruct_from_crops(local_features);

        let img_emb = self
            .vision_projector
            .forward((global_features, reconstructed)); // (expecting (729, 2048)
        img_emb
    }
}

pub struct VisionAttention {
    qkv: Linear,
    proj: Linear,
}
impl VisionAttention {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            qkv: Linear::new_permuted(VIS_DIM, VIS_DIM * 3, true, cx),
            proj: Linear::new_permuted(VIS_DIM, VIS_DIM, true, cx),
        }
    }
}

impl Module<GraphTensor> for VisionAttention {
    type Output = GraphTensor;
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

        let mut att = q.matmul(k.permute((0, 1, 3, 2)));

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
    type Output = GraphTensor;
    fn forward(&self, mut x: GraphTensor) -> Self::Output {
        let ln_x1 = self.ln1.forward(x);

        let l_attn = self.attn.forward(ln_x1);
        x = x + l_attn;

        let ln_x2 = self.ln2.forward(x);
        x = x + self.mlp.forward(ln_x2);

        x
    }
}

fn create_patches(mut x: GraphTensor) -> GraphTensor {
    let (b, c, h, w) = x.dims4();
    let p1 = VIS_PATCH_SIZE;

    x = x.reshape((b, c, h / p1, p1, w / p1, p1));

    x = x.permute((0, 2, 4, 1, 3, 5));

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
        let mut x = create_patches(img);

        x = self.linear.forward(x);

        x = x + self.pos_emb;

        for layer in 0..self.blocks.len() {
            x = self.blocks[layer].forward(x);
        }

        x = self.ln.forward(x);
        x
    }
}

pub fn reconstruct_from_crops(crops: GraphTensor) -> GraphTensor {
    let tiling_h = 378;
    let tiling_w = 378;
    let (num_crops, crop_height, crop_width, channels) = crops.dims4();

    let channels_usize = channels.to_usize().unwrap();
    let crop_height_usize = crop_height.to_usize().unwrap();
    let crop_width_usize = crop_width.to_usize().unwrap();
    let margin_pixels = 4 * 1;

    let output_h = (crop_height * margin_pixels) * tiling_h + 2 * margin_pixels;
    let output_w = (crop_width * margin_pixels) * tiling_w + 2 * margin_pixels;

    // this seems to be where the huge number of graph operations are coming from (1 slice)
    // need to find a better way to do this...
    // match the enumeration 1:1
    let mut reconstructed = None;
    for crop in 0..num_crops.to_usize().unwrap() {
        let tile_y = crop / tiling_w; // floor division for usizes
        let tile_x = crop % tiling_w;

        let mut x_start = margin_pixels;
        let mut x_end = crop_width_usize - margin_pixels;
        let mut y_start = margin_pixels;
        let mut y_end = crop_height_usize - margin_pixels;

        if tile_x == 0 {
            x_start = 0;
        }
        if tile_x == tiling_w - 1 {
            x_end = crop_width_usize;
        }
        if tile_y == 0 {
            y_start = 0;
        }
        if tile_y == tiling_h - 1 {
            y_end = crop_height_usize;
        }

        let cropped_val = crops
            .slice((crop..crop + 1, y_start..y_end, x_start..x_end, ..))
            .contiguous();
        println!("CROP: {:?}", cropped_val.dims());
        if crop == 0 {
            reconstructed = Some(cropped_val);
        } else {
            if let Some(ref mut recon) = reconstructed {
                *recon = recon.concat_along(cropped_val, 0);
            }
        }
    }
    let output = reconstructed
        .unwrap()
        .reshape((output_h, output_w, channels_usize))
        .contiguous();
    println!("FINAL: {:?}", output.dims());
    output
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

impl SerializeModule for VisionModel {
    fn serialize(&self, s: &mut Serializer) {
        s.module("model/vision", &self.vision_projector);
        s.module("model/vision", &self.vision_encoder);
    }
}

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

impl SerializeModule for VisionProjector {
    fn serialize(&self, s: &mut Serializer) {
        s.module("proj_mlp", &self.mlp);
        // no learned weights for avg pool
    }
}

impl SerializeModule for VisionAttention {
    fn serialize(&self, s: &mut Serializer) {
        s.module("qkv", &self.qkv);
        s.module("proj", &self.proj);
    }
}

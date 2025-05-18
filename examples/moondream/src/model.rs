use luminal::{hl_ops::binary::F32Pow, prelude::*};
use luminal_nn::*;

pub const TXT_N_HEADS: usize = 32;
pub const TXT_DIM: usize = 2048;
pub const TXT_HEAD_DIM: usize = TXT_DIM / TXT_N_HEADS;
pub const TXT_N_LAYERS: usize = 1; //24; keeping it truly simple for now
pub const TXT_VOCAB: usize = 9; //51_200;
pub const TXT_N_KV: usize = 32;
pub const TXT_FF_DIM: usize = 8192;

pub const VIS_DIM: usize = 1152;
pub const VIS_PATCH_SIZE: usize = 14;
pub const VIS_PATCH_DIM: usize = VIS_PATCH_SIZE * VIS_PATCH_SIZE * 3; // 588
pub const VIS_NUM_PATCHES: usize = 729;
pub const VIS_FF_DIM: usize = 4304;
pub const VIS_N_LAYERS: usize = 1; //27; keeping it truly simple for now
pub const VIS_ENC_N_HEADS: usize = 16;
pub const VIS_HEAD_DIM: usize = VIS_DIM / VIS_ENC_N_HEADS;

// guessing: (1) redo attention for encoding head as parameter

pub const DIFF_THRESHOLD: f32 = 1e-4;

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
        println!("A: {:?}\nV: {:?}", att.dims(), v.dims());

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

        x.diff("./bins/block_mid.bin", DIFF_THRESHOLD);

        let ln_x2 = self.ln2.forward(x);
        x = x + self.mlp.forward(ln_x2);

        x.diff("./bins/block_end.bin", DIFF_THRESHOLD); // broken
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
        let mut x = create_patches(img);
        x.diff("./bins/create_patches.bin", DIFF_THRESHOLD);
        x = self.linear.forward(x);
        x.diff("./bins/linear.bin", DIFF_THRESHOLD);
        self.pos_emb.diff("./bins/wpos_emb.bin", DIFF_THRESHOLD);
        println!("{:?} + {:?}", self.pos_emb.dims(), x.dims());
        x = x + self.pos_emb; // source of NaN
        println!("= {:?}", x.dims());
        x.diff("./bins/pos_emb.bin", DIFF_THRESHOLD);

        for layer in 0..self.blocks.len() {
            x = self.blocks[layer].forward(x);
        }
        x.diff("./bins/vis_attn.bin", DIFF_THRESHOLD);

        x = self.ln.forward(x);
        x
    }
}

pub struct SelfAttention {
    qkv: Linear,  // TXT_DIM → (n_heads + 2·n_kv) · head_dim
    proj: Linear, // TXT_DIM → TXT_DIM
}

impl SelfAttention {
    pub fn new(cx: &mut Graph) -> Self {
        //  q_dim = H·d   kv_dim = 2·Hkv·d
        const QKV_DIM: usize = TXT_DIM * (1 + 2 * TXT_N_KV / TXT_N_HEADS); // 2048 * (1+2) = 6144
        Self {
            qkv: Linear::new_permuted(TXT_DIM, QKV_DIM, true, cx),
            proj: Linear::new_permuted(TXT_DIM, TXT_DIM, false, cx),
        }
    }
}

fn apply_rotary_embeddings(x: GraphTensor, start: usize, end: usize) -> GraphTensor {
    // x.diff("./bins/x_input.bin", DIFF_THRESHOLD);

    let x_rot = x.slice((.., .., .., ..Expression::from(32))).contiguous();
    let x_pass = x.slice((.., .., .., Expression::from(32)..)).contiguous();

    // x_rot.diff("./bins/x_rot.bin", DIFF_THRESHOLD);
    // x_pass.diff("./bins/x_pass.bin", DIFF_THRESHOLD);

    // Get tensor dimensions
    let (b, h, s, d) = x_rot.dims4();
    let d_q = d / 2;
    println!("___ {:?} | {d_q}", x_rot.dims());

    // Split into real/imaginary components
    let xq_r = x_rot.slice((.., .., .., ..d_q)).contiguous();
    let xq_i = x_rot.slice((.., .., .., d_q..)).contiguous();
    // xq_r.diff("./bins/xq_r.bin", DIFF_THRESHOLD);
    // xq_i.diff("./bins/xq_i.bin", DIFF_THRESHOLD);

    // 1. Compute inverse frequencies (with proper float exponentiation)
    let mut freqs = 1.0
        / 10_000_f32
            .pow((x.graph().arange(d / 2) * 2) / d)
            .reshape((1, d / 2));
    // freqs.diff("./bins/freqs.bin", DIFF_THRESHOLD);
    let t = x.graph().arange(TXT_DIM).expand(1, 1);
    // t.diff("./bins/t.bin", DIFF_THRESHOLD);

    freqs = t.matmul(freqs); // keep as we will keep reusing this as a constant, like weights
                             // freqs.diff("./bins/freqs_mul.bin", DIFF_THRESHOLD);

    let cos = freqs
        .cos()
        .slice((Expression::from(start)..Expression::from(end + 1), ..))
        .reshape((1, (end + 1) - start, d / 2))
        .expand(1, 32)
        .contiguous();
    let sin = freqs
        .sin()
        .slice((Expression::from(start)..Expression::from(end + 1), ..))
        .reshape((1, (end + 1) - start, d / 2))
        .expand(1, 32)
        .contiguous();

    println!(
        "{:?} * {:?} - {:?} * {:?}",
        xq_r.dims(),
        cos.dims(),
        xq_i.dims(),
        sin.dims()
    );

    // cos.diff("./bins/freqs_cos.bin", DIFF_THRESHOLD);
    // sin.diff("./bins/freqs_sin.bin", DIFF_THRESHOLD);

    let xr = xq_r * cos - xq_i * sin;
    // xr.diff("./bins/xq_out_r.bin", DIFF_THRESHOLD);
    let xi: GraphTensor = xq_r * sin + xq_i * cos;
    // xi.diff("./bins/xq_out_i.bin", DIFF_THRESHOLD);

    println!("{:?} - {:?} ", xr.dims(), xi.dims());

    let (a, b, c, d) = xr.dims4();
    let x_rotated = xr
        .reshape((a, b, c, d, 1))
        .contiguous()
        .concat_along(xi.reshape((a, b, c, d, 1)).contiguous(), 4)
        .reshape((a, b, c, d * 2))
        .contiguous();
    // x_rotated.diff("./bins/xq_out.bin", DIFF_THRESHOLD);
    let output = x_rotated.concat_along(x_pass, 3);
    // output.diff("./bins/output.bin", DIFF_THRESHOLD);

    output
}

impl Module<(GraphTensor, KVCache)> for SelfAttention {
    type Output = (GraphTensor, KVCache);

    fn forward(&self, (x, (k_cache, v_cache)): (GraphTensor, KVCache)) -> Self::Output {
        let (b, s, _) = x.dims3();
        let (_, _, p, _) = k_cache.dims4();
        let head_dim = TXT_HEAD_DIM;

        let position_ids: Vec<usize> = vec![730, 731, 732, 733, 734, 735, 736, 737, 738]; // will be derived from p

        x.diff("./bins/attn_x.bin", DIFF_THRESHOLD);

        self.qkv
            .weight
            .diff("./bins/qvk_weight.bin", DIFF_THRESHOLD);
        self.qkv
            .bias
            .unwrap()
            .diff("./bins/qvk_bias.bin", DIFF_THRESHOLD);

        // fused projection
        let qkv = self.qkv.forward(x);

        println!(
            "{:?} = {} {} {} {}",
            qkv.shape, TXT_N_HEADS, TXT_HEAD_DIM, b, s
        );

        qkv.diff("./bins/qkv_out.bin", DIFF_THRESHOLD);
        let q_dim = TXT_N_HEADS * TXT_HEAD_DIM;
        let kv_dim = TXT_N_KV * TXT_HEAD_DIM;

        let q = qkv
            .slice((.., .., ..Expression::from(q_dim)))
            .contiguous()
            .reshape((b, s, TXT_N_HEADS, TXT_HEAD_DIM))
            .permute((0, 2, 1, 3));

        let k = qkv
            .slice((
                ..,
                ..,
                Expression::from(q_dim)..Expression::from(q_dim + kv_dim),
            ))
            .contiguous()
            .reshape((b, s, TXT_N_KV, TXT_HEAD_DIM))
            .permute((0, 2, 1, 3));

        let v = qkv
            .slice((.., .., Expression::from(q_dim + kv_dim)..))
            .contiguous()
            .reshape((b, s, TXT_N_KV, TXT_HEAD_DIM))
            .permute((0, 2, 1, 3));

        //confirming our pull is correct
        q.diff("./bins/q_b.bin", DIFF_THRESHOLD);
        k.diff("./bins/k_b.bin", DIFF_THRESHOLD);
        v.diff("./bins/v_b.bin", DIFF_THRESHOLD);

        // rotary & cache
        let q = apply_rotary_embeddings(q, position_ids[0], position_ids[position_ids.len() - 1]);
        let k = apply_rotary_embeddings(k, position_ids[0], position_ids[position_ids.len() - 1]);

        q.diff("./bins/q_rot.bin", DIFF_THRESHOLD);
        k.diff("./bins/k_rot.bin", DIFF_THRESHOLD);

        println!("{:?} x {:?} = attn", q.dims(), k.dims());

        // scaled dot product attention
        let mut att = q.matmul(k.permute((0, 1, 3, 2))); // dims are right, this is likely correct

        let mask = self
            .qkv
            .weight
            .graph()
            .triu(s, 1)
            .expand(0, 1)
            .expand(1, 32)
            .contiguous();
        println!("{:?}", mask.dims());
        let sqrt_dk = (head_dim as f32).sqrt();
        att = att * (1.0 / 8.0);
        att = att + (mask * f32::NEG_INFINITY);
        att = att.softmax(3);
        println!("A: {:?}\nV: {:?}", att.dims(), v.dims());

        let out = att.matmul(v).permute((0, 2, 1, 3)).reshape((b, s, TXT_DIM));
        out.diff("./bins/attn_out.bin", DIFF_THRESHOLD);

        // delcaring victory here, it won't match perfectly until the image stuff is done

        (self.proj.forward(out), (k.contiguous(), v.contiguous()))
    }
}

pub struct Mlp {
    pub fc1: Linear, // hidden -> intermediate
    pub fc2: Linear, // intermediate -> hidden
}

impl Mlp {
    pub fn new(hidden: usize, intermediate: usize, cx: &mut Graph) -> Self {
        Self {
            fc1: Linear::new_permuted(hidden, intermediate, false, cx),
            fc2: Linear::new_permuted(intermediate, hidden, false, cx),
        }
    }
    pub fn new_with_bias(hidden: usize, intermediate: usize, cx: &mut Graph) -> Self {
        Self {
            fc1: Linear::new_permuted(hidden, intermediate, true, cx),
            fc2: Linear::new_permuted(intermediate, hidden, true, cx),
        }
    }
}

impl Module<GraphTensor> for Mlp {
    type Output = GraphTensor;
    fn forward(&self, x: GraphTensor) -> Self::Output {
        self.fc2.forward(self.fc1.forward(x).gelu())
    }
}

pub struct TextBlock {
    ln: LayerNorm,
    attn: SelfAttention,
    mlp: Mlp,
}

impl TextBlock {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            ln: LayerNorm::new(TXT_DIM, true, true, true, 1e-5, cx),
            attn: SelfAttention::new(cx),
            mlp: Mlp::new(TXT_DIM, TXT_FF_DIM, cx),
        }
    }
}

impl Module<(GraphTensor, KVCache)> for TextBlock {
    type Output = (GraphTensor, KVCache);
    fn forward(&self, (x, cache): (GraphTensor, KVCache)) -> Self::Output {
        //layer norm
        let l_ln = self.ln.forward(x);
        self.ln
            .weight
            .unwrap()
            .diff("./bins/ln_weight.bin", DIFF_THRESHOLD);
        self.ln
            .bias
            .unwrap()
            .diff("./bins/ln_bias.bin", DIFF_THRESHOLD);
        l_ln.diff("./bins/ln.bin", DIFF_THRESHOLD);

        //attention
        let (l_attn, cache) = self.attn.forward((l_ln, cache));
        l_attn.diff("./bins/l_attn.bin", DIFF_THRESHOLD);

        //MLP
        let l_mlp = self.mlp.forward(l_ln);
        let y = x + l_attn + l_mlp;

        (y, cache)
    }
}

pub struct Moondream {
    vision_encoder: VisionEncoder,
    embed: Embedding,
    txt_blocks: Vec<TextBlock>,
}
impl Moondream {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            vision_encoder: VisionEncoder::new(cx),
            embed: Embedding::new(TXT_VOCAB, TXT_DIM, cx),
            txt_blocks: (0..TXT_N_LAYERS).map(|_| TextBlock::new(cx)).collect(),
        }
    }
}

pub type KVCache = (GraphTensor, GraphTensor);

impl Module<(GraphTensor, &[KVCache])> for Moondream {
    // Args: (image_tensor, token_ids, kv_cache[])
    type Output = (GraphTensor, Vec<KVCache>);
    fn forward(&self, (toks, cache): (GraphTensor, &[KVCache])) -> Self::Output {
        //_generate_text() pseudocode for now
        // toks should be all 1s of shape (1,9)

        // IMAGE BELOW:
        toks.diff("./bins/all_crops.bin", DIFF_THRESHOLD);
        let x = self.vision_encoder.forward(toks);
        x.diff("./bins/outputs.bin", DIFF_THRESHOLD);

        //TEXT BELOW:

        // toks.diff("./bins/prompt_tokens.bin", DIFF_THRESHOLD);

        // // prefill_prompt
        // let prompt_emb = self.embed.forward(toks);
        // prompt_emb.diff("./bins/prompt_emb.bin", DIFF_THRESHOLD);
        // prompt_emb.diff("./bins/ln_x.bin", DIFF_THRESHOLD);

        let new = Vec::with_capacity(TXT_N_LAYERS);

        // //text decoder block
        // let mut x = prompt_emb;
        // for layer in 0..self.txt_blocks.len() {
        //     let (y, cache) = self.txt_blocks[layer].forward((prompt_emb, cache[0]));
        //     x = y
        // }

        (x, new)
    }
}

///////////////////
/// Serializations
///

impl SerializeModule for Moondream {
    fn serialize(&self, s: &mut Serializer) {
        // Vision branch
        s.module("model/vision", &self.vision_encoder);
        // s.module("model/vision/proj_mlp", &self.vis_proj);
        // s.module("model/region", &self.region);

        // Text branch
        s.tensor("model/text/wte", self.embed.weight);
        for (i, blk) in self.txt_blocks.iter().enumerate() {
            s.module(&format!("model/text/blocks/{i}"), blk);
        }
        // s.module("model/text/post_ln", &self.txt_norm);
        // s.module("model/text/lm_head", &self.lm_head);
    }
}

impl SerializeModule for TextBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("ln", &self.ln);
        s.module("attn", &self.attn);
        s.module("mlp", &self.mlp);
    }
}

impl SerializeModule for SelfAttention {
    fn serialize(&self, s: &mut Serializer) {
        s.module("qkv", &self.qkv); // weight key: ".../qkv"
        s.module("proj", &self.proj); // weight key: ".../proj"
    }
}

impl SerializeModule for VisionAttention {
    fn serialize(&self, s: &mut Serializer) {
        s.module("qkv", &self.qkv); // weight key: ".../qkv"
        s.module("proj", &self.proj); // weight key: ".../proj"
    }
}

impl SerializeModule for Mlp {
    fn serialize(&self, s: &mut Serializer) {
        s.module("fc1", &self.fc1);
        s.module("fc2", &self.fc2);
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
impl SerializeModule for VisionBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("ln1", &self.ln1);
        s.module("ln2", &self.ln2);
        s.module("attn", &self.attn);
        s.module("mlp", &self.mlp);
    }
}

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

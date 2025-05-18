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

pub const DIFF_THRESHOLD: f32 = 1e-2;

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
    embed: Embedding,
    txt_blocks: Vec<TextBlock>,
}
impl Moondream {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
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
        toks.diff("./bins/prompt_tokens.bin", DIFF_THRESHOLD);

        // prefill_prompt
        let prompt_emb = self.embed.forward(toks);
        prompt_emb.diff("./bins/prompt_emb.bin", DIFF_THRESHOLD);
        prompt_emb.diff("./bins/ln_x.bin", DIFF_THRESHOLD);

        let new = Vec::with_capacity(TXT_N_LAYERS);

        //text decoder block
        let mut x = prompt_emb;
        for layer in 0..self.txt_blocks.len() {
            let (y, cache) = self.txt_blocks[layer].forward((prompt_emb, cache[0]));
            x = y
        }

        (x, new)
    }
}

///////////////////
/// Serializations
///

impl SerializeModule for Moondream {
    fn serialize(&self, s: &mut Serializer) {
        // Vision branch
        // s.module("model/vision", &self.vision);
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

impl SerializeModule for Mlp {
    fn serialize(&self, s: &mut Serializer) {
        s.module("fc1", &self.fc1);
        s.module("fc2", &self.fc2);
    }
}

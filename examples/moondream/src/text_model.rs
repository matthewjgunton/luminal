use luminal::{hl_ops::binary::F32Pow, prelude::*};
use luminal_nn::*;

use crate::model::KVCache;

pub const TXT_N_HEADS: usize = 32;
pub const TXT_DIM: usize = 2048;
pub const TXT_HEAD_DIM: usize = TXT_DIM / TXT_N_HEADS;
pub const TXT_N_LAYERS: usize = 24;
pub const TXT_VOCAB_SIZE: usize = 51_200;
pub const TXT_N_KV: usize = 32;
pub const TXT_FF_DIM: usize = 8192;

pub struct LmHead {
    post_ln: LayerNorm,
    lm_head: Linear,
}

impl LmHead {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            post_ln: LayerNorm::new(TXT_DIM, true, true, true, 1e-5, cx),
            lm_head: Linear::new(TXT_DIM, TXT_VOCAB_SIZE, true, cx),
        }
    }
}

impl Module<(GraphTensor)> for LmHead {
    type Output = (GraphTensor);
    fn forward(&self, x: GraphTensor) -> Self::Output {
        let (_b, t, _c) = x.dims3();
        let hidden_bc = x
            .slice((.., Expression::from(t)..Expression::from(t), ..))
            .contiguous();

        let l_ln = self.post_ln.forward(hidden_bc);
        let logits = self.lm_head.forward(l_ln);
        logits
    }
}

pub struct SelfAttention {
    qkv: Linear,
    proj: Linear,
}

impl SelfAttention {
    pub fn new(cx: &mut Graph) -> Self {
        const QKV_DIM: usize = TXT_DIM * (1 + 2 * TXT_N_KV / TXT_N_HEADS);
        Self {
            qkv: Linear::new_permuted(TXT_DIM, QKV_DIM, true, cx),
            proj: Linear::new_permuted(TXT_DIM, TXT_DIM, false, cx),
        }
    }
}

fn apply_rotary_embeddings(x: GraphTensor, start: usize, end: usize) -> GraphTensor {
    let x_rot = x.slice((.., .., .., ..Expression::from(32))).contiguous();
    let x_pass = x.slice((.., .., .., Expression::from(32)..)).contiguous();

    let (b, h, s, d) = x_rot.dims4();
    let d_q = d / 2;

    let xq_r = x_rot.slice((.., .., .., ..d_q)).contiguous();
    let xq_i = x_rot.slice((.., .., .., d_q..)).contiguous();

    let mut freqs = 1.0
        / 10_000_f32
            .pow((x.graph().arange(d / 2) * 2) / d)
            .reshape((1, d / 2));

    let t = x.graph().arange(TXT_DIM).expand(1, 1);

    freqs = t.matmul(freqs); // keep as we will keep reusing this as a constant, like weights

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

    let xr = xq_r * cos - xq_i * sin;
    let xi: GraphTensor = xq_r * sin + xq_i * cos;

    let (a, b, c, d) = xr.dims4();
    let x_rotated = xr
        .reshape((a, b, c, d, 1))
        .contiguous()
        .concat_along(xi.reshape((a, b, c, d, 1)).contiguous(), 4)
        .reshape((a, b, c, d * 2))
        .contiguous();

    let output = x_rotated.concat_along(x_pass, 3);

    output
}

impl Module<(GraphTensor, KVCache, usize, usize)> for SelfAttention {
    type Output = (GraphTensor, KVCache);

    fn forward(
        &self,
        (x, (k_cache, v_cache), pos_start, pos_end): (GraphTensor, KVCache, usize, usize),
    ) -> Self::Output {
        let (b, s, _) = x.dims3();
        let head_dim = TXT_HEAD_DIM;

        let qkv = self.qkv.forward(x);

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

        // rotary & cache
        let q = apply_rotary_embeddings(q, pos_start, pos_end);
        let k = apply_rotary_embeddings(k, pos_start, pos_end);
        let k = k_cache.concat_along(k, 2);
        let v = v_cache.concat_along(v, 2);

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
        let sqrt_dk = (head_dim as f32).sqrt();
        att = att * (1.0 / sqrt_dk);
        att = att + (mask * f32::NEG_INFINITY);
        att = att.softmax(3);

        let out = att.matmul(v).permute((0, 2, 1, 3)).reshape((b, s, TXT_DIM));

        (self.proj.forward(out), (k.contiguous(), v.contiguous()))
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

impl Module<(GraphTensor, KVCache, usize, usize)> for TextBlock {
    type Output = (GraphTensor, KVCache);
    fn forward(
        &self,
        (x, cache, pos_start, pos_end): (GraphTensor, KVCache, usize, usize),
    ) -> Self::Output {
        let l_ln = self.ln.forward(x);

        let (l_attn, cache) = self.attn.forward((l_ln, cache, pos_start, pos_end));

        let l_mlp = self.mlp.forward(l_ln);
        let y = x + l_attn + l_mlp;

        (y, cache)
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

impl SerializeModule for LmHead {
    fn serialize(&self, s: &mut Serializer) {
        s.module("ln", &self.post_ln);
        s.module("linear", &self.lm_head);
    }
}

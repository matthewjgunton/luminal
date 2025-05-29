use luminal::{hl_ops::binary::F32Pow, prelude::*};
use luminal_nn::*;

use crate::image_model::{self, *};
use crate::text_model::{self, *};
use crate::BOS_ID;

pub struct Moondream {
    pub vision_encoder: VisionEncoder,
    pub text_encoder: Embedding,
    pub text_decoder: Vec<TextBlock>,
    pub vision_projector: VisionProjector,
    pub lm_head: LmHead,
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

impl Module<(GraphTensor, &[KVCache], usize, usize)> for Moondream {
    type Output = (GraphTensor, Vec<KVCache>);
    fn forward(
        &self,
        (toks, cache, pos_start, pos_end): (GraphTensor, &[KVCache], usize, usize),
    ) -> Self::Output {
        let mut prompt = self.text_encoder.forward(toks);

        let mut new_caches = vec![];
        let mut new_cache: KVCache;
        for layer in 0..self.text_decoder.len() {
            let (y, new_cache) =
                self.text_decoder[layer].forward((prompt, cache[layer], pos_start, pos_end));
            new_caches.push(new_cache);
            prompt = y;
        }

        let logits = self.lm_head.forward(prompt);

        (logits, new_caches)
    }
}

pub fn _run_vision_encoder(
    moondream: &Moondream,
    img: GraphTensor,
    cx: &mut Graph,
    cache: &Vec<KVCache>,
) -> Vec<KVCache> {
    let x = moondream.vision_encoder.forward(img);

    let (a, _, _) = x.dims3();

    let global_features = x
        .slice((..Expression::from(0), .., ..))
        .reshape((VIS_NUM_PATCHES, VIS_DIM)) // unsqueeze effectively
        .contiguous();
    let local_features = x
        .slice((Expression::from(1).., .., ..))
        .reshape((a - 1, VIS_N_LAYERS, VIS_N_LAYERS, VIS_DIM)) // unsqueeze effectively
        .contiguous();
    let reconstructed = image_model::reconstruct_from_crops(local_features);

    let img_emb = moondream
        .vision_projector
        .forward((global_features, reconstructed)); // (expecting (729, 2048)

    let bos_tensor = cx.tensor((1, 1)).set(vec![BOS_ID]); // (expecting 1,1,2048)
    moondream.text_encoder.forward(bos_tensor);
    let mut prompt = img_emb
        .reshape((1, VIS_NUM_PATCHES, TXT_DIM))
        .contiguous()
        .concat_along(bos_tensor, 1)
        .contiguous();
    let (_, t, _) = prompt.dims3();

    let mut new_caches = vec![];
    for layer in 0..moondream.text_decoder.len() {
        let (y, new_cache) =
            moondream.text_decoder[layer].forward((prompt, cache[layer], 0, t.to_usize().unwrap()));
        new_caches.push(new_cache);
        prompt = y;
    }
    let (lastk, lastv) = new_caches.get(new_caches.len() - 1).unwrap();
    println!("HITHITHIT");
    lastk.diff("./bins/lastk.bin", ATOL, RTOL);
    lastv.diff("./bins/lastv.bin", ATOL, RTOL);
    (new_caches)
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

        // Text branch
        s.tensor("model/text/wte", self.text_encoder.weight);
        for (i, blk) in self.text_decoder.iter().enumerate() {
            s.module(&format!("model/text/blocks/{i}"), blk);
        }
        s.module("model/text", &self.lm_head);
    }
}

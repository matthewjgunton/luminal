use luminal::prelude::{binary::F32Pow, *};
use luminal_nn::{Conv2D, Embedding, LayerNorm, Linear};
use std::cmp::{max, min};

////////////////////////////////////////////////////////////////
// ethically sourced from https://huggingface.co/vikhyatk/moondream2/blob/main/config.py //
////////////////////////////////////////////////////////////////
pub const TXT_N_LAYERS: usize = 1;
pub const TXT_N_HEADS: usize = 10;
pub const TXT_HEAD_DIM: usize = 10;
pub const VIS_BASE_SIZE:usize = 378;
pub const PATCH_SIZE:usize = 14;
pub const OVERLAP_MARGIN:usize = 4;
pub const MAX_CROPS:usize = 12;

pub type KVCache = (GraphTensor, GraphTensor);

pub struct VisionEncoder {
    patch: (),
    pos: (),
    blks: (),
}
impl VisionEncoder {
    pub fn new(cx: &mut Graph) -> Self {
        // let patch = PatchEmbed::new(cx);
        // let n_patches = (VIS_CROP / VIS_PATCH).pow(2);
        Self {
            patch: (),
            pos: (),
            blks: (),
        }
    }
}

pub fn select_tiling(height: usize, width: usize, crop_size: usize, max_crops: usize) -> (usize, usize) {
    if height <= crop_size || width <= crop_size {
        return (1, 1);
    }

    let min_h = (height + crop_size - 1) / crop_size;
    let min_w = (width + crop_size - 1) / crop_size;

    if min_h * min_w > max_crops {
        let ratio = ((max_crops as f64) / (min_h * min_w) as f64).sqrt();
        return (
            max(1, (min_h as f64 * ratio).floor() as usize),
            max(1, (min_w as f64 * ratio).floor() as usize),
        );
    }

    let mut h_tiles = ((max_crops as f64 * height as f64 / width as f64).sqrt().floor()) as u32;
    let mut w_tiles = ((max_crops as f64 * width as f64 / height as f64).sqrt().floor()) as u32;

    h_tiles = max(h_tiles, min_h as u32);
    w_tiles = max(w_tiles, min_w as u32);

    if h_tiles * w_tiles > max_crops as u32 {
        if w_tiles > h_tiles {
            w_tiles = (max_crops as u32) / h_tiles;
        } else {
            h_tiles = (max_crops as u32) / w_tiles;
        }
    }

    (max(1, h_tiles as usize), max(1, w_tiles as usize))
}

pub fn overlap_crop_image(
    input: &GraphTensor,
    cx: &mut Graph,
    overlap_margin: u32,
) -> Result<(Vec<GraphTensor>, (u32, u32)), Box<dyn std::error::Error>> {

    let (b, c, h, w) = input.dims4(); // (1, C, H, W)
    assert_eq!(b, 1, "Batch size must be 1");

    let margin_pixels: usize = PATCH_SIZE * OVERLAP_MARGIN;
    let total_margin_pixels: usize = margin_pixels * 2;

    let crop_patches = VIS_BASE_SIZE / PATCH_SIZE;
    let crop_window_patches = crop_patches - 2 * OVERLAP_MARGIN;
    let crop_window_size = crop_window_patches * PATCH_SIZE;

    let (h_tiles, w_tiles) = select_tiling(
        h.to_usize().unwrap()  - total_margin_pixels,
        w.to_usize().unwrap() - total_margin_pixels,
        crop_window_size,
        MAX_CROPS,
    );

    //interpolation based off the goal size:


    let mut crops = Vec::new();




    // 1. Add global crop (just resize the original to base_size)
    let global = cx.tensor((1, c, base_size.0 as usize, base_size.1 as usize));
    cx.op_resize(input, &global)?;
    crops.push(global);

    // 2. Compute overlapping crops
    let crop_window_size = crop_window_size as usize;
    let base_h = base_size.0 as usize;
    let base_w = base_size.1 as usize;
    let margin = margin_pixels as usize;

    for i in 0..h_tiles {
        for j in 0..w_tiles {
            let y0 = i as usize * crop_window_size;
            let x0 = j as usize * crop_window_size;

            let y1 = (y0 + base_h).min(h);
            let x1 = (x0 + base_w).min(w);

            let crop = input.slice((.., .., y0..y1, x0..x1));
            let out = cx.tensor((1, c, base_h, base_w));
            cx.op_resize(&crop, &out)?;
            crops.push(out);
        }
    }

    Ok((crops, (h_tiles, w_tiles)))
}



impl Module<GraphTensor> for VisionEncoder {
    type Output = (); // (b, n_tokens, VIS_DIM)
    fn forward(&self, x: GraphTensor) -> Self::Output {
        // patch embedding first
        println!("\n\nvision encoder hit! | {:?}", x.shape);

        overlapped_crops = overlap_crop_image(&x, cx, overlap_margin, max_crops, base_size, patch_size)

        let normalizedCrops = overlapped_crops.transpose((0, 3, 1, 2)) / 255.0;
        normalizedCrops = (normalizedCrops - 0.5) / 0.5;
    }
}

pub struct Moondream {
    vision: VisionEncoder,
    vis_proj: (),
    region: (),
    embed: (),
    txt_blocks: Vec<()>,
    txt_norm: (),
    lm_head: (),
}

impl Moondream {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            vision: VisionEncoder::new(cx),
            vis_proj: (),
            region: (),
            embed: (),
            txt_blocks: [].to_vec(),
            txt_norm: (),
            lm_head: (),
        }
    }
}

impl Module<(GraphTensor, GraphTensor, &[KVCache])> for Moondream {
    // Args: (image_tensor, token_ids, kv_cache[])
    type Output = ((), ());
    fn forward(&self, (img, toks, cache): (GraphTensor, GraphTensor, &[KVCache])) -> Self::Output {
        println!("BEGUN");
        self.vision.forward(img);
        ((), ())
        // img.diff("./bins/image_encoder_input.bin", 1e-4); //MATCHED
        // let vis_tokens = self.vision.forward(img); // (b,n_vis,VIS_DIM)

        // let prefix = self.vis_proj.forward(vis_tokens);

        // let mut x = self.embed.forward(toks); // (b,seq,TXT_DIM)
        // x = prefix.concat_along(x, 1); // prepend vision prefix

        // // Transformer
        // let mut new = Vec::with_capacity(TXT_N_LAYERS);
        // for (blk, i) in self.txt_blocks.iter().zip(0..) {
        //     let (y, c) = blk.forward((x, cache[i]));
        //     x = y;
        //     new.push(c);
        // }

        // (self.lm_head.forward(self.txt_norm.forward(x)), new)
    }
}

impl SerializeModule for Moondream {
    fn serialize(&self, s: &mut Serializer) {
        // Vision branch
    }
}

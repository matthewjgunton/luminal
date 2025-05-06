use image::codecs::jpeg::JpegEncoder;
use image::{Rgb, RgbImage};
use std::error::Error;
use std::fs::File;

pub const VIS_DIM: usize = 1152;
pub const WIDTH: usize = 64;
pub const HEIGHT: usize = 64;

/// Turn your flat [1×H×W×VIS_DIM] Vec<f32> into an H×W RGB image,
/// pulling channels 0..3 and clamping [0,1]→[0,255].
fn tensor_to_rgb_image(flat: &[f32]) -> RgbImage {
    let mut img = RgbImage::new(WIDTH as u32, HEIGHT as u32);

    let to_u8 = |v: f32| {
        let iv = (v * 255.0).round() as i32;
        iv.clamp(0, 255) as u8
    };

    let width = WIDTH as usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let base = ((y * WIDTH + x) * VIS_DIM);
            let r = flat[base + 0];
            let g = flat[base + 1];
            let b = flat[base + 2];
            img.put_pixel(x as u32, y as u32, Rgb([to_u8(r), to_u8(g), to_u8(b)]));
        }
    }

    img
}

/// Save your tensor as a JPEG file at the given quality (1–100).
pub fn save_tensor_as_jpeg(flat: &[f32], path: &str, quality: u8) -> Result<(), Box<dyn Error>> {
    let img = tensor_to_rgb_image(flat);
    let fout = File::create(path)?;
    let mut encoder = JpegEncoder::new_with_quality(fout, quality);
    encoder.encode_image(&img)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // 1) Run your graph and retrieve the tensor:
    //
    //    let img_tensor = cx.constant(0f32)
    //        .expand_to((1, 64, 64, VIS_DIM))
    //        .retrieve();
    //    let flat = img_tensor.data();
    //
    // For this example, assume you already have `flat: Vec<f32>`:
    // equivalent of let img = cx.constant(0).expand_to((1, 64, 64, model::VIS_DIM));
    let flat: Vec<f32> = vec![0.0; (1 * WIDTH * HEIGHT * VIS_DIM) as usize];

    // 2) Write out example.jpg at 90% quality:
    save_tensor_as_jpeg(&flat, "example.jpg", 90)?;
    println!("Wrote example.jpg ({}×{})", WIDTH, HEIGHT);
    Ok(())
}

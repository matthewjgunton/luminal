// use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
// use itertools::Itertools;
// use std::cmp::{max, min};
// use std::path::Path;

// use luminal::prelude::*;

// use crate::model::{self, DIFF_THRESHOLD};

// pub const IMG_MAX_CROPS: usize = 12;
// pub const IMG_MAX_SIZE: usize = 378;
// pub const IMG_OVERLAP_MARGIN: usize = 4;

// pub fn prepare_crops(image_path: &Path, cx: &mut Graph) -> (GraphTensor, (i32, i32)) {
//     let image = image::open(image_path).unwrap();

//     // Get image dimensions
//     let (width, height) = image.dimensions();
//     let width = width as usize;
//     let height = height as usize;
//     let channels = 3; // RGB

//     // Convert to RGB8 and extract raw pixel data
//     let rgb_image = image.to_rgb8();
//     let raw_pixels = rgb_image.into_raw();

//     // Convert u8 pixels to f32 in HWC format
//     let image_data: Vec<f32> = raw_pixels.into_iter().map(|pixel| pixel as f32).collect();

//     // Verify the data size matches expectations
//     assert_eq!(image_data.len(), width * height * channels);

//     // For debugging - you can remove this line once you're confident it works
//     println!(
//         "Loaded image: {}x{}x{}, {} pixels",
//         width,
//         height,
//         channels,
//         image_data.len()
//     );

//     diff("./bins/img_data.bin", &image_data, DIFF_THRESHOLD);

//     // let input = vec![1 as f32; 767 * 773 * 3];
//     // diff("./bins/nd_input.bin", &image_data, model::DIFF_THRESHOLD);

//     let (mut all_crops, tiling_tuple) = overlap_crop_image(image_data, cx, 767, 773, 3);

//     all_crops = all_crops.permute((0, 3, 1, 2));
//     all_crops = all_crops / 255.0;
//     all_crops = all_crops - 0.5;
//     all_crops = all_crops / 0.5;

//     (all_crops, tiling_tuple)
// }

// pub fn overlap_crop_image(
//     image: Vec<f32>,
//     cx: &mut Graph,
//     original_h: usize,
//     original_w: usize,
//     img_channels: usize,
// ) -> (GraphTensor, (i32, i32)) {
//     // Convert margin from patch units to pixels
//     let margin_pixels = model::VIS_PATCH_SIZE * IMG_OVERLAP_MARGIN;
//     let total_margin_pixels = margin_pixels * 2; // Both sides

//     // Calculate crop parameters
//     let crop_patches = IMG_MAX_SIZE / model::VIS_PATCH_SIZE; // patches per crop dimension
//     let crop_window_patches = crop_patches - (2 * IMG_OVERLAP_MARGIN); // usable patches
//     let crop_window_size = crop_window_patches * model::VIS_PATCH_SIZE; // usable size in pixels

//     let (tile_height, tile_width) =
//         select_tiling(original_h, original_w, crop_window_size, IMG_MAX_CROPS);

//     let n_crops = tile_height * tile_width + 1;
//     // crops[crop_idx * IMG_MAX_SIZE * IMG_MAX_SIZE * channels + h * IMG_MAX_SIZE * channels + w * channels + c]
//     let mut crops = vec![0.0_f32; n_crops * IMG_MAX_SIZE * IMG_MAX_SIZE * img_channels];
//     // diff("./bins/crops_1.bin", &crops, model::DIFF_THRESHOLD);

//     let target_height = tile_height * crop_window_size + total_margin_pixels;
//     let target_width = tile_width * crop_window_size + total_margin_pixels;
//     // diff("./bins/nd_input.bin", &image, model::DIFF_THRESHOLD);
//     // Initialize the SimpleImage
//     let vips_image = SimpleImage::new_from_array(image, original_h, original_w, img_channels);

//     // Calculate scales for target dimensions
//     let scale_x = target_width as f32 / original_w as f32;
//     let scale_y = target_height as f32 / original_h as f32;

//     // Resize to target dimensions
//     let resized = vips_image.resize(scale_x, scale_y);

//     // Get the actual resized dimensions
//     let (resized_height, resized_width, _) = resized.dimensions();

//     // Global crop
//     let global_scale_x = IMG_MAX_SIZE as f32 / original_w as f32;
//     let global_scale_y = IMG_MAX_SIZE as f32 / original_h as f32;
//     let global_vips = vips_image.resize(global_scale_x as f32, global_scale_y as f32);
//     let global_data = global_vips.numpy();
//     // diff(
//     //     "./bins/global_data.bin",
//     //     &global_data,
//     //     model::DIFF_THRESHOLD,
//     // );

//     // Copy global crop to crops[0]
//     let (global_height, global_width, _) = global_vips.dimensions();
//     for h in 0..global_height {
//         for w in 0..global_width {
//             for c in 0..img_channels {
//                 // Calculate source index (in global_data)
//                 let src_idx = (h * global_width + w) * img_channels + c;

//                 // Calculate destination index (in crops[0])
//                 let dst_idx =
//                     (0 * IMG_MAX_SIZE * IMG_MAX_SIZE + h * IMG_MAX_SIZE + w) * img_channels + c;

//                 // Make sure we don't go out of bounds
//                 if src_idx < global_data.len() {
//                     crops[dst_idx] = global_data[src_idx];
//                 }
//             }
//         }
//     }

//     // Access the resized image data
//     let resized_data = resized.numpy();

//     // For the tiled crops
//     for h in 0..tile_height {
//         for w in 0..tile_width {
//             let y0 = h * crop_window_size;
//             let x0 = w * crop_window_size;

//             // Calculate actual crop size considering image boundaries
//             let y_end = min(y0 + IMG_MAX_SIZE, resized_height);
//             let x_end = min(x0 + IMG_MAX_SIZE, resized_width);

//             // Calculate crop region dimensions
//             let crop_height = y_end - y0;
//             let crop_width = x_end - x0;

//             // Calculate the crop index (1 + h * tile_width + w) - global crop is at index 0
//             let crop_idx = 1 + h * tile_width + w;

//             // Extract the crop region and copy it to the crops array
//             for cy in 0..crop_height {
//                 for cx in 0..crop_width {
//                     for c in 0..img_channels {
//                         // Source index in the resized image
//                         let src_idx = ((y0 + cy) * resized_width + (x0 + cx)) * img_channels + c;

//                         // Destination index in the crops array
//                         let dst_idx =
//                             (crop_idx * IMG_MAX_SIZE * IMG_MAX_SIZE + cy * IMG_MAX_SIZE + cx)
//                                 * img_channels
//                                 + c;

//                         // Copy the pixel (with bounds check)
//                         if src_idx < resized_data.len() {
//                             crops[dst_idx] = resized_data[src_idx];
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     // Convert the crops Vec into a GraphTensor
//     diff("./bins/overlap_crops.bin", &crops, model::DIFF_THRESHOLD);

//     let crops_tensor = cx
//         .tensor((n_crops, IMG_MAX_SIZE, IMG_MAX_SIZE, img_channels))
//         .set(crops);

//     (crops_tensor, (tile_height as i32, tile_width as i32))
// }

// pub fn select_tiling(
//     height: usize,
//     width: usize,
//     crop_size: usize,
//     max_crops: usize,
// ) -> (usize, usize) {
//     if height <= crop_size || width <= crop_size {
//         return (1, 1);
//     }

//     // Minimum required tiles in each dimension
//     let min_h = (height as f32 / crop_size as f32).ceil() as usize;
//     let min_w = (width as f32 / crop_size as f32).ceil() as usize;

//     // If minimum required tiles exceed max_crops, return proportional distribution
//     if min_h * min_w > max_crops {
//         let ratio = ((max_crops as f64) / (min_h * min_w) as f64).sqrt();
//         return (
//             max(1, ((min_h as f64) * ratio).floor() as i32) as usize,
//             max(1, ((min_w as f64) * ratio).floor() as i32) as usize,
//         );
//     }

//     // Perfect aspect-ratio tiles that satisfy max_crops
//     let h_tiles = ((max_crops as f64 * height as f64 / width as f64).sqrt()).floor() as i32;
//     let w_tiles = ((max_crops as f64 * width as f64 / height as f64).sqrt()).floor() as i32;

//     // Ensure we meet minimum tile requirements
//     let mut h_tiles = max(h_tiles as usize, min_h);
//     let mut w_tiles = max(w_tiles as usize, min_w);

//     // If we exceeded max_crops, scale down the larger dimension
//     if h_tiles * w_tiles > max_crops {
//         if w_tiles > h_tiles {
//             w_tiles = max_crops / h_tiles;
//         } else {
//             h_tiles = max_crops / w_tiles;
//         }
//     }

//     (max(1, h_tiles), max(1, w_tiles))
// }

// /// A simple image representation that can be resized
// struct SimpleImage {
//     data: Vec<f32>,
//     width: usize,
//     height: usize,
//     channels: usize,
// }

// impl SimpleImage {
//     /// Create a new SimpleImage from raw data
//     fn new(data: Vec<f32>, width: usize, height: usize, channels: usize) -> Self {
//         SimpleImage {
//             data,
//             width,
//             height,
//             channels,
//         }
//     }

//     /// Create a SimpleImage from a 3D array
//     fn new_from_array(data: Vec<f32>, height: usize, width: usize, channels: usize) -> Self {
//         // Copy data to ensure we have ownership
//         Self::new(data, width, height, channels)
//     }

//     /// Resize the image using bilinear interpolation
//     fn resize(&self, scale_x: f32, vscale: f32) -> Self {
//         let new_width = (self.width as f32 * scale_x as f32).round() as usize;
//         let new_height = (self.height as f32 * vscale as f32).round() as usize;

//         let mut new_data = vec![0 as f32; new_width * new_height * self.channels];

//         // Simple bilinear interpolation
//         for y in 0..new_height {
//             for x in 0..new_width {
//                 // Map back to original coordinates
//                 let src_x = (x as f32 / scale_x).min(self.width as f32 - 1.0);
//                 let src_y = (y as f32 / vscale).min(self.height as f32 - 1.0);

//                 let src_x_floor = src_x.floor() as usize;
//                 let src_y_floor = src_y.floor() as usize;
//                 let src_x_ceil = min(src_x_floor + 1, self.width - 1);
//                 let src_y_ceil = min(src_y_floor + 1, self.height - 1);

//                 let x_weight = src_x - src_x_floor as f32;
//                 let y_weight = src_y - src_y_floor as f32;

//                 for c in 0..self.channels {
//                     // Get the four neighboring pixels
//                     let top_left = self.get_pixel_value(src_x_floor, src_y_floor, c);
//                     let top_right = self.get_pixel_value(src_x_ceil, src_y_floor, c);
//                     let bottom_left = self.get_pixel_value(src_x_floor, src_y_ceil, c);
//                     let bottom_right = self.get_pixel_value(src_x_ceil, src_y_ceil, c);

//                     // Bilinear interpolation
//                     let top = top_left * (1.0 - x_weight) + top_right * x_weight;
//                     let bottom = bottom_left * (1.0 - x_weight) + bottom_right * x_weight;
//                     let pixel_value = top * (1.0 - y_weight) + bottom * y_weight;

//                     new_data[(y * new_width + x) * self.channels + c] = pixel_value;
//                 }
//             }
//         }

//         SimpleImage::new(new_data, new_width, new_height, self.channels)
//     }

//     /// Get a pixel value at the specified location and channel
//     fn get_pixel_value(&self, x: usize, y: usize, channel: usize) -> f32 {
//         let index = (y * self.width + x) * self.channels + channel;
//         self.data[index] as f32
//     }

//     /// Convert the image to a numpy-like array
//     fn numpy(&self) -> Vec<f32> {
//         self.data.clone()
//     }

//     /// Get image dimensions
//     fn dimensions(&self) -> (usize, usize, usize) {
//         (self.height, self.width, self.channels)
//     }
// }

// /// Compare a vector of f32 values with values from a binary file
// ///
// /// # Arguments
// /// * `file_path` - Path to the binary file containing f32 values
// /// * `data` - Vector of f32 values to compare with the file
// /// * `threshold` - Maximum allowed difference between corresponding values
// ///
// /// # Returns
// /// * `bool` - true if the data matches within threshold, false otherwise
// pub fn diff(file_path: impl Into<std::path::PathBuf>, data: &Vec<f32>, threshold: f32) -> bool {
//     use colored::*;
//     use std::fs;
//     use std::path::PathBuf;

//     let path = file_path.into();

//     // Read binary file and convert chunks of 4 bytes to f32
//     let bin_data = match fs::read(&path) {
//         Ok(bytes) => bytes
//             .chunks(4)
//             .map(|i| f32::from_ne_bytes([i[0], i[1], i[2], i[3]]).clamp(f32::MIN, f32::MAX))
//             .collect::<Vec<_>>(),
//         Err(e) => {
//             println!(
//                 "{}",
//                 format!("Error reading file {:?}: {}", path, e).bold().red()
//             );
//             return false;
//         }
//     };

//     // Check for length mismatch
//     if data.len() != bin_data.len() {
//         println!(
//             "{}",
//             format!(
//                 "{} | Length mismatch! Data: {}, File: {}",
//                 path.as_os_str().to_str().unwrap_or("Unknown path"),
//                 data.len(),
//                 bin_data.len()
//             )
//             .bold()
//             .red()
//         );
//         return false;
//     }

//     // Check for NaN values
//     let data_nan = data.iter().any(|i| i.is_nan());
//     let file_nan = bin_data.iter().any(|i| i.is_nan());

//     if data_nan {
//         println!(
//             "{}",
//             format!(
//                 "{} | Data contains NaN!",
//                 path.to_str().unwrap_or("Unknown path")
//             )
//             .bold()
//             .red()
//         );
//     }

//     if file_nan {
//         println!(
//             "{}",
//             format!(
//                 "{} | File contains NaN!",
//                 path.to_str().unwrap_or("Unknown path")
//             )
//             .bold()
//             .red()
//         );
//     }

//     if data_nan || file_nan {
//         return false;
//     }

//     // Compare values
//     let mut matched = true;
//     for (i, (a, b)) in data.iter().zip(bin_data.iter()).enumerate() {
//         if (a - b).abs() > threshold {
//             println!(
//                 "{}",
//                 format!(
//                     "{} | Value Mismatch!",
//                     path.to_str().unwrap_or("Unknown path")
//                 )
//                 .bold()
//                 .red()
//             );

//             if let Some((i, _)) = data.iter().enumerate().find(|(_, i)| i.is_nan()) {
//                 println!("Index {} is NaN!", i.to_string().bold());
//             }

//             println!("{a} is not equal to {b}, index {i}");

//             // Calculate statistics
//             let avg_dist = data
//                 .iter()
//                 .zip(bin_data.iter())
//                 .map(|(a, b)| (a - b).abs())
//                 .sum::<f32>()
//                 / data.len() as f32;

//             let max_dist = data
//                 .iter()
//                 .zip(bin_data.iter())
//                 .map(|(a, b)| (a - b).abs())
//                 .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
//                 .unwrap_or(f32::NAN);

//             let sum_dist = data
//                 .iter()
//                 .zip(bin_data.iter())
//                 .map(|(a, b)| (a - b) * (a - b))
//                 .sum::<f32>();

//             println!(
//                 "Avg dist: {}, Max dist: {} Sum dist: {}",
//                 avg_dist.to_string().bold().red(),
//                 max_dist.to_string().bold().red(),
//                 sum_dist.to_string().bold().red(),
//             );

//             // Show sample data
//             println!("{}: {:?}", "This".bold(), &data[..data.len().min(10)]);
//             println!(
//                 "{}: {:?}",
//                 "File".bold(),
//                 &bin_data[..bin_data.len().min(10)]
//             );

//             // Show largest mismatches
//             println!(
//                 "Largest Mismatches: {:?}",
//                 data.iter()
//                     .zip(bin_data.iter())
//                     .filter(|(a, b)| (**a - **b).abs() > 0.01)
//                     .sorted_by(|(a, b), (c, d)| (**c - **d)
//                         .abs()
//                         .partial_cmp(&(**a - **b).abs())
//                         .unwrap_or(std::cmp::Ordering::Equal))
//                     .take(10)
//                     .collect::<Vec<_>>()
//             );

//             // Show statistics
//             println!(
//                 "A avg: {} B avg: {}",
//                 data.iter().sum::<f32>() / data.len() as f32,
//                 bin_data.iter().sum::<f32>() / bin_data.len() as f32
//             );

//             println!(
//                 "A max: {} B max: {}",
//                 data.iter()
//                     .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
//                     .unwrap_or(&f32::NAN),
//                 bin_data
//                     .iter()
//                     .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
//                     .unwrap_or(&f32::NAN)
//             );

//             println!(
//                 "A min: {} B min: {}",
//                 data.iter()
//                     .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
//                     .unwrap_or(&f32::NAN),
//                 bin_data
//                     .iter()
//                     .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
//                     .unwrap_or(&f32::NAN)
//             );

//             matched = false;
//             break;
//         }
//     }

//     if matched {
//         println!(
//             "{}",
//             format!("{} matched", path.to_str().unwrap_or("Unknown path"))
//                 .bold()
//                 .bright_green()
//         );
//     }

//     matched
// }

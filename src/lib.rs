pub mod color_recode;
pub mod colorspace;
pub mod dct;
pub mod qim;

use bitvec::prelude::{BitVec, Lsb0};
use image::{DynamicImage, GenericImageView};

/// Uses DCT together with QIM-DM to embed the colored watermark image into the host image
///
/// Higher `step_size` generally yields better extraction result, but might reduce the imperceptability of the watermark
///
/// Panics if the host image is not 512 * 512 or the watermark image is not 128 * 128
pub fn embed_watermark(
    host_image: &str,
    watermark_image: &str,
    key: u64,
    step_size: f32,
) -> DynamicImage {
    let host = image::open(host_image).expect("Failed to open host image");
    let (h_width, h_height) = host.dimensions();
    assert!(
        h_width == 512 && h_height == 512,
        "{}",
        format!(
            "Host image:'{}' must be 512 * 512, got {} * {}",
            host_image, h_width, h_height
        )
    );

    // Convert the image to YCbCr colorspace
    let (mut y_plane, cb_plane, cr_plane) = colorspace::convert_to_YCbCr(&host);

    // Split Y plane into 8 * 8 blocks for DCT operation
    let mut y_blocks = dct::split_into_blocks(&mut y_plane, h_width as usize, h_height as usize);

    // DCT on Y blocks
    dct::apply_2d_dct(&mut y_blocks);

    let wm = image::open(watermark_image).expect("Failed to open watermark image");
    let (wm_width, wm_height) = wm.dimensions();
    assert!(wm_width == 128 && wm_height == 128);

    // Recoding the watermark
    let wm_bits = color_recode::recode_to_3bits(&wm);

    // QIM-DM to embed the watermark with the preset key and step_size
    let dithers = qim::generate_dither_signal(12, step_size, key);
    for (i, bits) in wm_bits.chunks(12).enumerate() {
        qim::embed_wm(&mut y_blocks[i], &bits.to_bitvec(), &dithers, step_size);
    }

    // IDCT on watermarked Y blocks
    dct::apply_2d_idct(&mut y_blocks);

    // Convert Y blocks back to Y plane
    let watermarked_y_plane = dct::merge_into_plane(&y_blocks, h_width as usize, h_height as usize);

    // Convert back to RGB colorspace and return the RGB DynamicImage
    colorspace::convert_to_RGB(
        h_width,
        h_height,
        &watermarked_y_plane,
        &cb_plane,
        &cr_plane,
    )
}

/// Extract the colored watermark embedded using DCT + QIM-DM watermarking scheme
///
/// Returns the original bit stream and the reconstructed RGB DynamicImage
///
/// Works with images of size 512 * 512 and watermark of size 128 * 128,
/// with watermark embedded in implementation specific locations
pub fn extract_watermark(
    watermarked_image: &str,
    key: u64,
    step_size: f32,
) -> (BitVec, DynamicImage) {
    let wmkd_image = image::open(watermarked_image).unwrap();
    let (width, height) = wmkd_image.dimensions();

    // Convert the watermarked image to YCbCr colorspace and DCT on Y blocks
    let (mut wmkd_y_plane, _, _) = colorspace::convert_to_YCbCr(&wmkd_image);

    let mut wmkd_y_blocks =
        dct::split_into_blocks(&mut wmkd_y_plane, width as usize, height as usize);

    dct::apply_2d_dct(&mut wmkd_y_blocks);

    // Extract the watermark from each block
    let dithers = qim::generate_dither_signal(12, step_size, key);
    let mut extracted_wm: BitVec<usize, Lsb0> = BitVec::new();
    for block in wmkd_y_blocks.iter() {
        let tmp = qim::extract_wm(block, &dithers, step_size);
        for bit in tmp {
            extracted_wm.push(bit);
        }
    }

    // Reconstruct the image from bits and save the recovered watermark
    let reconstructed_wm_image = color_recode::recode_to_rgb(&extracted_wm, 128, 128);
    (extracted_wm, reconstructed_wm_image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    const INPUT_DIR: &str = "/tmp/color_watermark/assets";
    const OUTPUT_DIR: &str = "/tmp/color_watermark/output";

    #[test]
    fn test_3bit_recodification() {
        let wm = image::open(format!("{}/wm_img1.png", INPUT_DIR)).unwrap();
        let (width, height) = wm.dimensions();

        color_recode::recode_to_rgb(&color_recode::recode_to_3bits(&wm), width, height)
            .save(format!("{}/wm_img1_recoded.png", OUTPUT_DIR))
            .unwrap();

        let wm = image::open(format!("{}/wm_img2.png", INPUT_DIR)).unwrap();
        let (w, h) = wm.dimensions();
        color_recode::recode_to_rgb(&color_recode::recode_to_3bits(&wm), w, h)
            .save(format!("{}/wm_img2_recoded.png", OUTPUT_DIR))
            .unwrap();
    }

    #[test]
    fn test_2d_dct() {
        let image_path = format!("{}/pepper.tiff", INPUT_DIR);
        let image = image::open(image_path).unwrap();
        let (width, height) = image.dimensions();

        let (mut blocks_r, mut blocks_g, mut blocks_b) = dct::split_image_into_blocks(&image);

        dct::apply_2d_dct(&mut blocks_r);
        dct::apply_2d_dct(&mut blocks_g);
        dct::apply_2d_dct(&mut blocks_b);

        let transformed_image =
            dct::reconstruct_image_from_rgb(&blocks_r, &blocks_g, &blocks_b, width, height);
        transformed_image
            .save(format!("{}/pepper_2d_dct.png", OUTPUT_DIR))
            .expect("Failed to save image");

        dct::apply_2d_idct(&mut blocks_r);
        dct::apply_2d_idct(&mut blocks_g);
        dct::apply_2d_idct(&mut blocks_b);

        let unchanged_image =
            dct::reconstruct_image_from_rgb(&blocks_r, &blocks_g, &blocks_b, width, height);

        unchanged_image
            .save(format!("{}/pepper_unchanged_dct.png", OUTPUT_DIR))
            .expect("Failed to save unchanged_image");
    }

    #[test]
    fn test_rgb_toforth_ycrcb() {
        let image_path = format!("{}/pepper.tiff", INPUT_DIR);
        let image = image::open(image_path).unwrap();
        let (width, height) = image.dimensions();

        let (y_plane, cb_plane, cr_plane) = colorspace::convert_to_YCbCr(&image);

        let rgb_img = colorspace::convert_to_RGB(
            width,
            height,
            y_plane.as_slice(),
            cb_plane.as_slice(),
            cr_plane.as_slice(),
        );

        rgb_img
            .save(format!("{}/pepper_unchanged_color.png", OUTPUT_DIR))
            .unwrap();
    }

    #[test]
    fn test_complete_workflow() {
        let key = 0123456_u64;
        let step_size = 100.0;

        // *********** Embedding the watermark **********
        let image_path = format!("{}/pepper.tiff", INPUT_DIR);
        let image = image::open(image_path).unwrap();
        let (width, height) = image.dimensions();

        // Convert the image to YCbCr colorspace
        let (mut y_plane, cb_plane, cr_plane) = colorspace::convert_to_YCbCr(&image);

        // Split Y plane into 8 * 8 blocks for DCT operation
        let mut y_blocks = dct::split_into_blocks(&mut y_plane, width as usize, height as usize);

        // DCT on Y blocks
        dct::apply_2d_dct(&mut y_blocks);

        // Load the watermark image
        let wm_path = format!("{}/wm_img1.png", INPUT_DIR);
        let wm_image = image::open(wm_path).unwrap();
        // Recoding the watermark
        let wm_bits = color_recode::recode_to_3bits(&wm_image);

        // QIM-DM to embed the watermark with the preset key and step_size
        let dithers = qim::generate_dither_signal(12, step_size, key);
        for (i, bits) in wm_bits.chunks(12).enumerate() {
            qim::embed_wm(&mut y_blocks[i], &bits.to_bitvec(), &dithers, step_size);
        }

        // In between embedding result test
        let mut extracted_wm: BitVec<usize, Lsb0> = BitVec::new();
        for block in y_blocks.iter() {
            let tmp = qim::extract_wm(&block, &dithers, step_size);
            for bit in tmp {
                extracted_wm.push(bit);
            }
        }
        color_recode::recode_to_rgb(&extracted_wm, 128, 128)
            .save(format!("{}/in_between.png", OUTPUT_DIR))
            .unwrap();

        // IDCT on watermarked Y blocks
        dct::apply_2d_idct(&mut y_blocks);

        // Convert Y blocks back to Y plane
        let watermarked_y_plane = dct::merge_into_plane(&y_blocks, width as usize, height as usize);

        // Convert back to RGB colorspace
        let wmd_image =
            colorspace::convert_to_RGB(width, height, &watermarked_y_plane, &cb_plane, &cr_plane);

        // Save the watermarked image
        wmd_image
            .save(format!("{}/watermarked_img.png", OUTPUT_DIR))
            .unwrap();

        // ************ Extracting the watermark ***************
        let wmkd_image = image::open(format!("{}/watermarked_img.png", OUTPUT_DIR)).unwrap();
        let (width, height) = wmkd_image.dimensions();

        // Convert the watermarked image to YCbCr colorspace and DCT on Y blocks
        let (mut wmkd_y_plane, _, _) = colorspace::convert_to_YCbCr(&wmkd_image);

        let mut wmkd_y_blocks =
            dct::split_into_blocks(&mut wmkd_y_plane, width as usize, height as usize);

        dct::apply_2d_dct(&mut wmkd_y_blocks);

        // Extract the watermark from each block
        let mut extracted_wm: BitVec<usize, Lsb0> = BitVec::new();
        for block in wmkd_y_blocks.iter() {
            let tmp = qim::extract_wm(&block, &dithers, step_size);
            for bit in tmp {
                extracted_wm.push(bit);
            }
        }

        // Reconstruct the image from bits and save the recovered watermark
        let reconstructed_wm = color_recode::recode_to_rgb(&extracted_wm, 128, 128);
        reconstructed_wm
            .save(format!("{}/reconstructed_wm.png", OUTPUT_DIR))
            .unwrap();
    }

    #[test]
    fn test_interface() {
        let k = 2143658709;
        for i in vec![1, 2] {
            let wm_path = format!("{}/wm_img{}.png", INPUT_DIR, i);
            for ss in vec![10.0, 20.0, 50.0, 100.0] {
                for image in std::fs::read_dir(INPUT_DIR).unwrap() {
                    let image = image.unwrap();
                    let image_path = image.path();

                    if let Some(ext) = image_path.extension() {
                        if ext == "tiff" {
                            println!(
                                "Embedding {} into {} with step_size {}",
                                wm_path,
                                image_path.to_string_lossy(),
                                ss as u32
                            );
                            let wmkd_image_path = format!(
                                "{}/embed_extract{}/{}_{}_wmkd_image.png",
                                OUTPUT_DIR,
                                i,
                                ss as u32,
                                image_path.file_stem().unwrap().to_string_lossy()
                            );
                            let output_file = std::path::Path::new(&wmkd_image_path);
                            if !output_file.exists() {
                                std::fs::create_dir_all(output_file.parent().unwrap()).unwrap();
                            }
                            let wmkd_img =
                                embed_watermark(&image_path.to_str().unwrap(), &wm_path, k, ss);
                            wmkd_img.save(&wmkd_image_path).unwrap();

                            println!("Extracting watermark from {}", wmkd_image_path);
                            let (_, extracted_wm) = extract_watermark(&wmkd_image_path, k, ss);
                            let extracted_wm_path = format!(
                                "{}/embed_extract{}/{}_{}_extracted_wm.png",
                                OUTPUT_DIR,
                                i,
                                ss as u32,
                                image_path.file_stem().unwrap().to_string_lossy()
                            );
                            extracted_wm.save(&extracted_wm_path).unwrap();
                        }
                    }
                }
            }
        }
    }
}

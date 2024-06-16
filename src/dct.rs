use rustdct::DctPlanner;
use image::{DynamicImage, GenericImageView};

pub const BLOCK_WIDTH: usize = 8;

/// Splits a dynamic image into 8 * 8 blocks
/// 
/// Returns (r, g, b) in form of Vec of (Vec of 64 * f32)
pub fn split_image_into_blocks(image: &DynamicImage) -> 
    (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (width, height) = image.dimensions();
    println!("Processing a {} * {} image", width, height);

    let mut blocks_r = Vec::new();
    let mut blocks_g = Vec::new();
    let mut blocks_b = Vec::new();

    // ordering by y then x to flush less cache
    for y in (0..height).step_by(8) {
        for x in (0..width).step_by(8) {
            let mut block_r = Vec::new();
            let mut block_g = Vec::new();
            let mut block_b = Vec::new();

            for j in 0..8 {
                for i in 0..8 {
                    let pixel = image.get_pixel(x + i, y + j).0;
                    block_r.push(pixel[0] as f32);
                    block_g.push(pixel[1] as f32);
                    block_b.push(pixel[2] as f32);
                }
            }

            blocks_r.push(block_r);
            blocks_g.push(block_g);
            blocks_b.push(block_b);
        }
    }

    (blocks_r, blocks_g, blocks_b)
}

/// Splits a color plane into 8 * 8 blocks
pub fn split_into_blocks(plane: &mut Vec<u8>, width: usize, height: usize) -> Vec<Vec<f32>> {
    let mut blocks = Vec::new();

    for y in (0..height).step_by(BLOCK_WIDTH) {
        for x in (0..width).step_by(BLOCK_WIDTH) {
            let mut block = Vec::new();

            for j in 0..BLOCK_WIDTH {
                for i in 0..BLOCK_WIDTH {
                    block.push(plane[(y + j) * width + (x + i)] as f32);
                }
            }

            blocks.push(block);
        }
    }

    blocks
}

/// Merge a Vec of 8 * 8 blocks back to a color plane
pub fn merge_into_plane(blocks: &Vec<Vec<f32>>, width: usize, height: usize) -> Vec<u8> {
    let mut plane = vec![0_u8; width * height];

    for (block_idx, block) in blocks.iter().enumerate() {
        let x = (block_idx % (width / BLOCK_WIDTH)) * BLOCK_WIDTH;
        let y = (block_idx / (width / BLOCK_WIDTH)) * BLOCK_WIDTH;

        for j in 0..BLOCK_WIDTH {
            for i in 0..BLOCK_WIDTH {
                plane[(y + j) * width + (x + i)] = block[j * BLOCK_WIDTH + i] as u8;
            }
        }
    }

    plane
}

/// Applies 2D DCT2 on a Vec of 8 * 8 blocks
/// 
/// Changes are made in-place
pub fn apply_2d_dct(blocks: &mut Vec<Vec<f32>>) {
    let mut planner = DctPlanner::new();
    let dct = planner.plan_dct2(8);

    for block in blocks.iter_mut() {
        // Apply DCT to each row
        for row in block.chunks_mut(8) {
            dct.process_dct2(row);
        }

        // Transpose the block
        let mut transposed_block = vec![0f32; 64];
        for i in 0..8 {
            for j in 0..8 {
                transposed_block[i * 8 + j] = block[j * 8 + i];
            }
        }

        // Apply DCT to each column (which are now rows of the transposed block)
        for row in transposed_block.chunks_mut(8) {
            dct.process_dct2(row);
        }

        // Transpose the block back to its original orientation
        for i in 0..8 {
            for j in 0..8 {
                block[j * 8 + i] = transposed_block[i * 8 + j];
            }
        }
    }
}


/// Applies 2D DCT3 (IDCT) on a Vec of 8 * 8 blocks
/// 
/// Changes are made in-place
pub fn apply_2d_idct(blocks: &mut Vec<Vec<f32>>) {
    let mut planner = DctPlanner::new();
    let idct = planner.plan_dct3(8);

    for block in blocks.iter_mut() {
        // Apply IDCT to each row
        for row in block.chunks_mut(8) {
            idct.process_dct3(row);
        }

        // Transpose the block
        let mut transposed_block = vec![0f32; 64];
        for i in 0..8 {
            for j in 0..8 {
                transposed_block[i * 8 + j] = block[j * 8 + i];
            }
        }

        // Apply IDCT to each column (which are now rows of the transposed block)
        for row in transposed_block.chunks_mut(8) {
            idct.process_dct3(row);
        }

        // Transpose the block back to its original orientation
        // and apply the normalization coefficient along the way, 4 / (height * width)
        let coeff = 4.0 / (8.0 * 8.0);
        for i in 0..8 {
            for j in 0..8 {
                block[j * 8 + i] = transposed_block[i * 8 + j] * coeff;
            }
        }
    }
}


use image::{DynamicImage, GenericImage, GenericImageView};
use yuvutils_rs::{
    BufferStoreMut, YuvConversionMode, YuvPlanarImage, YuvPlanarImageMut, YuvRange,
    YuvStandardMatrix,
};

/// Takes an RGB DynamicImage and convert to YCrCb
///
/// Return value: `(y_plane, cb_plane, cr_plane)`
#[allow(non_snake_case)]
pub fn convert_to_YCbCr(image: &DynamicImage) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (width, height) = image.dimensions();
    println!("convert_to_YCbCr image dimensions: {}x{}", width, height);

    let buffer_size = (width * height) as usize;
    let mut y: Vec<u8> = vec![0_u8; buffer_size];
    let mut cr: Vec<u8> = vec![0_u8; buffer_size];
    let mut cb: Vec<u8> = vec![0_u8; buffer_size];

    let y_plane = BufferStoreMut::Borrowed(y.as_mut_slice());
    let u_plane = BufferStoreMut::Borrowed(cb.as_mut_slice());
    let v_plane = BufferStoreMut::Borrowed(cr.as_mut_slice());

    // => RGB8
    let rgb_image = image.to_rgb8();
    let rgb = rgb_image.as_raw();
    let (width, height) = image.dimensions();
    let (rgb_stride, y_stride, cb_stride, cr_stride) = get_strides(width, false);

    let mut planar = YuvPlanarImageMut {
        y_plane,
        y_stride,
        u_plane,
        u_stride: cb_stride,
        v_plane,
        v_stride: cr_stride,
        width,
        height,
    };

    yuvutils_rs::rgb_to_yuv444(
        &mut planar,
        rgb,
        rgb_stride,
        YuvRange::Full,
        YuvStandardMatrix::Bt709,
        YuvConversionMode::Balanced,
    )
    .unwrap();

    (y, cb, cr)
}

/// Convert YCrCb to RGB DynamicImage
#[allow(non_snake_case)]
pub fn convert_to_RGB(
    width: u32,
    height: u32,
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
) -> DynamicImage {
    let (rgb_stride, y_stride, cb_stride, cr_stride) = get_strides(width, false);
    let mut rgb = vec![0_u8; (width * height * 3) as usize];

    let planar = YuvPlanarImage {
        y_plane,
        y_stride,
        u_plane: cb_plane,
        u_stride: cb_stride,
        v_plane: cr_plane,
        v_stride: cr_stride,
        width,
        height,
    };
    yuvutils_rs::yuv444_to_rgb(
        &planar,
        rgb.as_mut_slice(),
        rgb_stride,
        YuvRange::Full,
        YuvStandardMatrix::Bt709,
    )
    .unwrap();

    let mut img = DynamicImage::new_rgb8(width, height);

    for x in 0..height {
        for y in 0..width {
            let r = rgb[(x * width * 3 + y * 3) as usize];
            let g = rgb[(x * width * 3 + y * 3 + 1) as usize];
            let b = rgb[(x * width * 3 + y * 3 + 2) as usize];
            let a = 255;
            img.put_pixel(y, x, image::Rgba([r, g, b, a]));
        }
    }

    img
}

/// Calculates and returns the strides needed for colorspace conversion
///
/// Return value: `(rgb_stride, y_stride, cb_stride, cr_stride)`
///
/// set downsample to true when using 422 conversion, false when using 444
fn get_strides(width: u32, downsample: bool) -> (u32, u32, u32, u32) {
    let rgb_stride = width * 3; // 3 bytes per pixel for RGB
    let y_stride = width; // 1 byte per pixel for Y
    let cb_stride = if downsample { (width + 1) / 2 } else { width }; // subsampled horizontally
    let cr_stride = if downsample { (width + 1) / 2 } else { width }; // subsampled horizontally

    (rgb_stride, y_stride, cb_stride, cr_stride)
}

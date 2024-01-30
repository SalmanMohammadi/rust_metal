extern crate nalgebra as na;
use std::mem;
use metal::{BufferRef, ComputeCommandEncoderRef, ComputePipelineState, MTLOrigin, MTLRegion, MTLSize};

use na::SMatrix;
use metal::{Device, Function, DeviceRef, MTLResourceOptions, Buffer, Texture, TextureRef, TextureDescriptor};

fn empty_texture(size : u64, device: &DeviceRef) -> Texture {
    let tx = TextureDescriptor::new();
    tx.set_width(size);
    tx.set_height(size);
    tx.set_pixel_format(metal::MTLPixelFormat::R32Uint);
    let texture = device.new_texture(&tx);
    texture
}
fn matrix_to_texture<const R: usize>(
    m: &SMatrix<u32, R, R>, 
    device: &DeviceRef) -> Texture {
    let tx = TextureDescriptor::new();
    tx.set_width(R as u64);
    tx.set_height(R as u64);
    tx.set_pixel_format(metal::MTLPixelFormat::R32Uint);
    let texture = device.new_texture(&tx);
    let region = MTLRegion { 
        origin: MTLOrigin { x: 0, y: 0, z: 0 }, 
        size: MTLSize { width: R as u64, height: R as u64, depth: 1 }
    };
    let stride = R * std::mem::size_of::<u32>();
    texture.replace_region(
        region, 
        0, 
        unsafe { std::mem::transmute(m.as_ptr()) }, 
        stride as u64);
    texture
}

fn matrix_to_buffer<const R: usize, const C:usize>(
    m: &SMatrix<u32, R, C>, 
    device: &DeviceRef) -> (Buffer, u64) {
    let size = m.ncols() as u64 * std::mem::size_of::<u32>() as u64;
    (device.new_buffer_with_data(
        unsafe { std::mem::transmute(m.as_ptr()) },
        size * m.nrows() as u64,
        MTLResourceOptions::StorageModeShared,
    ), size)
}

fn set_pipieline_state(
    device: &DeviceRef,
    compute_encoder: &ComputeCommandEncoderRef, 
    function: Function) -> ComputePipelineState{
    let pipeline = device.new_compute_pipeline_state_with_function(&function).unwrap();
    compute_encoder.set_compute_pipeline_state(&pipeline);
    pipeline
}

fn encode_textures(
    compute_encoder: &ComputeCommandEncoderRef, 
    textures: &[Option<&TextureRef>; 3]) {
    // for (i, texture) in textures.iter().enumerate() {
    //         compute_encoder.set_texture(i as u64, *texture);
    //     }
// }
    compute_encoder.set_textures(
        0,
        textures
    );
    }
fn encode_buffers(
    compute_encoder: &ComputeCommandEncoderRef, 
    buffers: &[Option<&BufferRef>; 3]) {
    let offsets = vec![0u64; buffers.len()];
    compute_encoder.set_buffers(
        0,
        buffers,
        &offsets
    );
}

fn main() {
    type Matrix3x3u32 = SMatrix<u32, 8, 8>;

    // setup metal device, command buffers, and compute encoders... 
    let device: &DeviceRef = &Device::system_default().expect("No device found");
    const LIB_DATA: &[u8] = include_bytes!("../metal/linalg.metallib");
    let lib = device.new_library_with_data(LIB_DATA).unwrap();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let compute_encoder = command_buffer.new_compute_command_encoder();
    let m1 = Matrix3x3u32::repeat(2);
    let m2 = Matrix3x3u32::repeat(2);
 
    let metal_add = lib.get_function("add", None).unwrap();

    let tx_a = matrix_to_texture(&m1, device);
    let tx_b = matrix_to_texture(&m2, device);
    let tx_c = empty_texture(tx_a.width(), device);
    
    let pipeline:ComputePipelineState = set_pipieline_state(device, compute_encoder, metal_add);
    encode_textures(compute_encoder, &[Some(&tx_a), Some(&tx_b), Some(&tx_c)]);
    let n: u64 = m1.ncols() as u64;


    let threadgroup_size = MTLSize { 
        width: 16, 
        height: 16, 
        depth: 1 
    };
    // let threadgroup_width = pipeline.thread_execution_width();
    // let threadgroup_height = pipeline.max_total_threads_per_threadgroup() / threadgroup_width;
    let threadgroup_count = MTLSize { 
        width: (n + threadgroup_size.width - 1) / threadgroup_size.width, 
        height: (n + threadgroup_size.height - 1) / threadgroup_size.height, 
        depth: 1
    };
    compute_encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
    compute_encoder.end_encoding();
    command_buffer.commit();

    command_buffer.wait_until_completed();
    let mut data = vec![0u32; (n * n).try_into().unwrap()];

    let region = MTLRegion { 
        origin: MTLOrigin { x: 0, y: 0, z: 0 }, 
        size: MTLSize { width: n as u64, height: n as u64, depth: 1 }
    };

    let size_of_u32 = mem::size_of::<u32>() as u64;
    tx_c.get_bytes(
        data.as_mut_ptr() as _,
        (n * size_of_u32) as u64,
        region,
        0
    );
    let matrix = Matrix3x3u32::from_row_slice(&data);
    assert_eq!(matrix, m1 + m2, "Metal addition should equal nalgebra addition!");
}

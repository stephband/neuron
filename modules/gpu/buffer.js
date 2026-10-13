/**
 * GPU Buffer Helpers
 * 
 * Provides memory management for WebGPU storage, uniform, and staging buffers.
 */

/**
 * Creates a GPU Storage Buffer (usable as read/write storage in compute shaders).
 * Flags: STORAGE | COPY_DST (can upload to it) | COPY_SRC (can copy out of it for readback).
 */
export function createStorageBuffer(device, sizeOrData, label = 'storage_buffer') {
  let byteSize;
  let initialData = null;

  if (sizeOrData instanceof Float32Array || Array.isArray(sizeOrData)) {
    initialData = sizeOrData instanceof Float32Array ? sizeOrData : new Float32Array(sizeOrData);
    byteSize = initialData.byteLength;
  } else if (typeof sizeOrData === 'number') {
    byteSize = sizeOrData;
  } else {
    throw new Error('sizeOrData must be a byte length number or a Float32Array');
  }

  // Ensure minimum size and 4-byte alignment
  byteSize = Math.max(16, (byteSize + 3) & ~3);

  const buffer = device.createBuffer({
    label,
    size: byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  if (initialData) {
    device.queue.writeBuffer(buffer, 0, initialData.buffer, initialData.byteOffset, initialData.byteLength);
  }

  return buffer;
}

/**
 * Creates a GPU Uniform Buffer (read-only parameters like matrix dimensions M, N, K).
 * Uniform buffers must be a multiple of 16 bytes.
 */
export function createUniformBuffer(device, data, label = 'uniform_buffer') {
  const view = ArrayBuffer.isView(data) ? data : new Uint32Array(data);
  // Uniform buffers require sizes aligned to 16 bytes
  const alignedBytes = Math.max(16, Math.ceil(view.byteLength / 16) * 16);

  const buffer = device.createBuffer({
    label,
    size: alignedBytes,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(buffer, 0, view.buffer, view.byteOffset, view.byteLength);
  return buffer;
}

/**
 * Updates a GPU buffer with new CPU data.
 */
export function uploadToBuffer(device, gpuBuffer, float32Data, byteOffset = 0) {
  const data = float32Data instanceof Float32Array ? float32Data : new Float32Array(float32Data);
  device.queue.writeBuffer(gpuBuffer, byteOffset, data.buffer, data.byteOffset, data.byteLength);
}

/**
 * Reads back data from a GPU buffer into a new Float32Array.
 * Uses a temporary staging buffer with MAP_READ.
 */
export async function readFromBuffer(device, gpuBuffer, byteLength) {
  // Staging buffer requires COPY_DST | MAP_READ
  const stagingBuffer = device.createBuffer({
    label: 'staging_read_buffer',
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder({ label: 'readback_encoder' });
  encoder.copyBufferToBuffer(gpuBuffer, 0, stagingBuffer, 0, byteLength);
  device.queue.submit([encoder.finish()]);

  // Wait for GPU to finish copy and map the staging buffer
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const copyArrayBuffer = stagingBuffer.getMappedRange(0, byteLength);
  const result = new Float32Array(copyArrayBuffer.slice(0));

  stagingBuffer.unmap();
  stagingBuffer.destroy();

  return result;
}

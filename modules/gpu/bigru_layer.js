import { createStorageBuffer } from './buffer.js';

/**
 * GpuBidirectionalGru: WebGPU-accelerated Recurrent Layer
 * 
 * Executes full sequential bidirectional GRU inference natively on GPU VRAM:
 * 1. Parallel Pre-projection GEMM across all time steps: U = X * W_ih^T + b_ih
 * 2. Sequential recurrent forward pass (t = 0 -> T-1)
 * 3. Sequential recurrent backward pass (t = T-1 -> 0)
 * 4. Output concatenation H_t = [h_fwd_t ; h_bwd_t]
 */
export class GpuBidirectionalGru {
  constructor(device, gpuOps, inDim, hiddenDim, bidirectional = true) {
    this.device = device;
    this.gpuOps = gpuOps;
    this.inDim = inDim;
    this.hiddenDim = hiddenDim;
    this.bidirectional = bidirectional;
    this.outDim = bidirectional ? 2 * hiddenDim : hiddenDim;

    const H = hiddenDim;

    // GPU Buffers for Forward Cell Weights
    this.bufW_ih_fwd = createStorageBuffer(device, 3 * H * inDim * 4, 'gru_Wih_fwd');
    this.bufB_ih_fwd = createStorageBuffer(device, 3 * H * 4, 'gru_bih_fwd');
    this.bufW_hh_fwd = createStorageBuffer(device, 3 * H * H * 4, 'gru_Whh_fwd');
    this.bufB_hh_fwd = createStorageBuffer(device, 3 * H * 4, 'gru_bhh_fwd');

    // GPU Buffers for Backward Cell Weights (if bidirectional)
    if (bidirectional) {
      this.bufW_ih_bwd = createStorageBuffer(device, 3 * H * inDim * 4, 'gru_Wih_bwd');
      this.bufB_ih_bwd = createStorageBuffer(device, 3 * H * 4, 'gru_bih_bwd');
      this.bufW_hh_bwd = createStorageBuffer(device, 3 * H * H * 4, 'gru_Whh_bwd');
      this.bufB_hh_bwd = createStorageBuffer(device, 3 * H * 4, 'gru_bhh_bwd');
    }
  }

  /**
   * Upload weights from CPU BidirectionalGru or raw Float32Arrays
   */
  loadWeights(weights) {
    const queue = this.device.queue;

    const wih_fwd = weights.Wih_fwd instanceof Float32Array ? weights.Wih_fwd : new Float32Array(weights.Wih_fwd);
    const bih_fwd = weights.bih_fwd instanceof Float32Array ? weights.bih_fwd : new Float32Array(weights.bih_fwd);
    const whh_fwd = weights.Whh_fwd instanceof Float32Array ? weights.Whh_fwd : new Float32Array(weights.Whh_fwd);
    const bhh_fwd = weights.bhh_fwd instanceof Float32Array ? weights.bhh_fwd : new Float32Array(weights.bhh_fwd);

    queue.writeBuffer(this.bufW_ih_fwd, 0, wih_fwd.buffer, wih_fwd.byteOffset, wih_fwd.byteLength);
    queue.writeBuffer(this.bufB_ih_fwd, 0, bih_fwd.buffer, bih_fwd.byteOffset, bih_fwd.byteLength);
    queue.writeBuffer(this.bufW_hh_fwd, 0, whh_fwd.buffer, whh_fwd.byteOffset, whh_fwd.byteLength);
    queue.writeBuffer(this.bufB_hh_fwd, 0, bhh_fwd.buffer, bhh_fwd.byteOffset, bhh_fwd.byteLength);

    if (this.bidirectional && weights.Wih_bwd) {
      const wih_bwd = weights.Wih_bwd instanceof Float32Array ? weights.Wih_bwd : new Float32Array(weights.Wih_bwd);
      const bih_bwd = weights.bih_bwd instanceof Float32Array ? weights.bih_bwd : new Float32Array(weights.bih_bwd);
      const whh_bwd = weights.Whh_bwd instanceof Float32Array ? weights.Whh_bwd : new Float32Array(weights.Whh_bwd);
      const bhh_bwd = weights.bhh_bwd instanceof Float32Array ? weights.bhh_bwd : new Float32Array(weights.bhh_bwd);

      queue.writeBuffer(this.bufW_ih_bwd, 0, wih_bwd.buffer, wih_bwd.byteOffset, wih_bwd.byteLength);
      queue.writeBuffer(this.bufB_ih_bwd, 0, bih_bwd.buffer, bih_bwd.byteOffset, bih_bwd.byteLength);
      queue.writeBuffer(this.bufW_hh_bwd, 0, whh_bwd.buffer, whh_bwd.byteOffset, whh_bwd.byteLength);
      queue.writeBuffer(this.bufB_hh_bwd, 0, bhh_bwd.buffer, bhh_bwd.byteOffset, bhh_bwd.byteLength);
    }
  }

  /**
   * Sync from CPU BidirectionalGru layer instance
   */
  syncFromCpuLayer(cpuLayer) {
    const weights = {
      Wih_fwd: cpuLayer.cellFwd.W_ih.data,
      bih_fwd: cpuLayer.cellFwd.b_ih,
      Whh_fwd: cpuLayer.cellFwd.W_hh.data,
      bhh_fwd: cpuLayer.cellFwd.b_hh,
    };
    if (this.bidirectional && cpuLayer.cellBwd) {
      weights.Wih_bwd = cpuLayer.cellBwd.W_ih.data;
      weights.bih_bwd = cpuLayer.cellBwd.b_ih;
      weights.Whh_bwd = cpuLayer.cellBwd.W_hh.data;
      weights.bhh_bwd = cpuLayer.cellBwd.b_hh;
    }
    this.loadWeights(weights);
  }

  /**
   * Forward pass across sequence of length T on GPU
   * 
   * @param {GPUBuffer} bufX - Input features buffer [T x inDim]
   * @param {number} T - Sequence length
   * @param {GPUBuffer|null} bufH0 - Optional initial state [H]
   * @param {GPUCommandEncoder|null} encoder - Optional command encoder
   * @returns {GPUBuffer} bufOutH - Output hidden states [T x outDim]
   */
  forward(bufX, T, bufH0 = null, encoder = null) {
    const isLocalEncoder = !encoder;
    const cmdEncoder = encoder || this.device.createCommandEncoder({ label: 'gpu_bigru_forward_encoder' });

    const H = this.hiddenDim;
    const inDim = this.inDim;
    const outD = this.outDim;

    // Temporary intermediate buffers
    const bufU_mat_fwd = createStorageBuffer(this.device, T * 3 * H * 4, 'temp_U_mat_fwd');
    const bufU_fwd = createStorageBuffer(this.device, T * 3 * H * 4, 'temp_U_fwd');
    const bufOutH = createStorageBuffer(this.device, T * outD * 4, 'gru_outH');

    // 1. Forward Input Pre-projection: U_fwd = X * W_ih_fwd^T + b_ih_fwd
    this.gpuOps.dispatchMatmul(bufX, this.bufW_ih_fwd, bufU_mat_fwd, T, 3 * H, inDim, false, true, cmdEncoder);
    this.gpuOps.dispatchAddBias(bufU_mat_fwd, this.bufB_ih_fwd, bufU_fwd, T, 3 * H, cmdEncoder);

    // 2. Forward Recurrent Sequence Execution (t = 0 -> T-1)
    this.gpuOps.dispatchGruSequence(
      bufU_fwd,
      this.bufW_hh_fwd,
      this.bufB_hh_fwd,
      bufH0,
      bufOutH,
      T,
      H,
      outD,
      0, // offset 0
      0, // direction forward
      cmdEncoder
    );

    // 3. Backward Recurrent Sequence Execution (if bidirectional)
    if (this.bidirectional) {
      const bufU_mat_bwd = createStorageBuffer(this.device, T * 3 * H * 4, 'temp_U_mat_bwd');
      const bufU_bwd = createStorageBuffer(this.device, T * 3 * H * 4, 'temp_U_bwd');

      this.gpuOps.dispatchMatmul(bufX, this.bufW_ih_bwd, bufU_mat_bwd, T, 3 * H, inDim, false, true, cmdEncoder);
      this.gpuOps.dispatchAddBias(bufU_mat_bwd, this.bufB_ih_bwd, bufU_bwd, T, 3 * H, cmdEncoder);

      this.gpuOps.dispatchGruSequence(
        bufU_bwd,
        this.bufW_hh_bwd,
        this.bufB_hh_bwd,
        null, // Backward pass starts at zero
        bufOutH,
        T,
        H,
        outD,
        H, // offset H
        1, // direction backward
        cmdEncoder
      );
    }

    if (isLocalEncoder) {
      this.device.queue.submit([cmdEncoder.finish()]);
    }

    return bufOutH;
  }
}

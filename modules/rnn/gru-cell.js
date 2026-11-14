import { Matrix2D } from '../matrix.js';

/**
 * Standard Sigmoid activation: sigma(x) = 1 / (1 + exp(-x))
 */
export function sigmoid(x) {
  return 1.0 / (1.0 + Math.exp(-Math.max(-85, Math.min(85, x))));
}

/**
 * GRU Cell (Gated Recurrent Unit)
 * 
 * Computes:
 *   r_t = sigmoid(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_hr)   (Reset gate)
 *   z_t = sigmoid(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_hz)   (Update gate)
 *   n_t = tanh(W_in * x_t + b_in + r_t * (W_hn * h_{t-1} + b_hn)) (Candidate state)
 *   h_t = (1 - z_t) * h_{t-1} + z_t * n_t                     (New hidden state)
 * 
 * Weights are packed:
 *   W_ih: [3*H x inDim] (rows 0..H-1: r, H..2H-1: z, 2H..3H-1: n)
 *   W_hh: [3*H x H]
 *   b_ih: [3*H]
 *   b_hh: [3*H]
 */
export class GruCell {
  constructor(inDim, hiddenDim) {
    this.inDim = inDim;
    this.hiddenDim = hiddenDim;
    const H = hiddenDim;

    // Initialize weights with Xavier / He scaling
    this.W_ih = Matrix2D.randomXavier(3 * H, inDim);
    this.W_hh = Matrix2D.randomXavier(3 * H, H);
    this.b_ih = new Float32Array(3 * H);
    this.b_hh = new Float32Array(3 * H);

    // Gradients
    this.grad_W_ih = Matrix2D.zeros(3 * H, inDim);
    this.grad_W_hh = Matrix2D.zeros(3 * H, H);
    this.grad_b_ih = new Float32Array(3 * H);
    this.grad_b_hh = new Float32Array(3 * H);

    // Reusable sequence workspace buffers (grown on demand for T)
    this.capT = 0;
    this.bufU = null;
    this.bufGH = null;
    this.bufR = null;
    this.bufZ = null;
    this.bufN = null;
    this.bufHPrev = null;
    this.bufDU = null;
    this.bufDX = null;
    this.bufDGH = new Float32Array(3 * H);
    this.bufDHPrev = new Float32Array(H);
    this.bufDHNext = new Float32Array(H);
  }

  /**
   * Ensures reusable workspace buffers can hold at least T timesteps.
   */
  ensureCapacity(T) {
    if (T <= this.capT) return;
    const H = this.hiddenDim;
    this.capT = T;
    this.bufU = new Float32Array(T * 3 * H);
    this.bufGH = new Float32Array(T * 3 * H);
    this.bufR = new Float32Array(T * H);
    this.bufZ = new Float32Array(T * H);
    this.bufN = new Float32Array(T * H);
    this.bufHPrev = new Float32Array(T * H);
    this.bufDU = new Float32Array(T * 3 * H);
    this.bufDX = new Float32Array(T * this.inDim);
  }

  /**
   * Resets accumulated gradients to zero
   */
  zeroGrad() {
    this.grad_W_ih.fill(0);
    this.grad_W_hh.fill(0);
    this.grad_b_ih.fill(0);
    this.grad_b_hh.fill(0);
  }

  /**
   * Pre-projects an entire sequence of inputs X [T x inDim] into input gates [T x 3*H]
   * U = X * W_ih^T + b_ih
   */
  projectInputSequence(X, T, outU = null) {
    const H = this.hiddenDim;
    const U = outU || new Float32Array(T * 3 * H);
    const wData = this.W_ih.data;
    const bData = this.b_ih;
    const inD = this.inDim;

    for (let t = 0; t < T; t++) {
      const xOffset = t * inD;
      const uOffset = t * 3 * H;

      for (let i = 0; i < 3 * H; i++) {
        let sum = bData[i];
        const wRowOffset = i * inD;
        for (let j = 0; j < inD; j++) {
          sum += wData[wRowOffset + j] * X[xOffset + j];
        }
        U[uOffset + i] = sum;
      }
    }
    return U;
  }

  /**
   * Zero-allocation sequence forward pass across T timesteps (direction 0 = fwd, 1 = bwd),
   * writing directly into outH [T x outD] at column offset outOffset.
   */
  forwardSequence(X, T, h0, outH, outD, outOffset, reverse = false) {
    this.ensureCapacity(T);
    const H = this.hiddenDim;
    const U = this.projectInputSequence(X, T, this.bufU);
    const gH = this.bufGH;
    const rBuf = this.bufR;
    const zBuf = this.bufZ;
    const nBuf = this.bufN;
    const hPrevBuf = this.bufHPrev;
    const wData = this.W_hh.data;
    const bData = this.b_hh;

    const hPrev = this.bufDHPrev;
    if (h0 && !reverse) {
      for (let j = 0; j < H; j++) hPrev[j] = h0[j];
    } else {
      hPrev.fill(0);
    }

    for (let s = 0; s < T; s++) {
      const t = reverse ? (T - 1 - s) : s;
      const hOffset = t * H;
      const uOffset = t * 3 * H;
      const outRow = t * outD + outOffset;

      for (let j = 0; j < H; j++) {
        hPrevBuf[hOffset + j] = hPrev[j];
      }

      for (let i = 0; i < 3 * H; i++) {
        let sum = bData[i];
        const wRow = i * H;
        for (let j = 0; j < H; j++) {
          sum += wData[wRow + j] * hPrev[j];
        }
        gH[uOffset + i] = sum;
      }

      for (let j = 0; j < H; j++) {
        const r = sigmoid(U[uOffset + j] + gH[uOffset + j]);
        const z = sigmoid(U[uOffset + H + j] + gH[uOffset + H + j]);
        const n = Math.tanh(U[uOffset + 2 * H + j] + r * gH[uOffset + 2 * H + j]);
        const hNew = (1.0 - z) * hPrev[j] + z * n;

        rBuf[hOffset + j] = r;
        zBuf[hOffset + j] = z;
        nBuf[hOffset + j] = n;
        hPrev[j] = hNew;
        outH[outRow + j] = hNew;
      }
    }
  }

  /**
   * Zero-allocation sequence BPTT across T timesteps, reading cached buffers from forwardSequence.
   */
  backwardSequence(dH, X, T, outD, outOffset, reverse = false) {
    const H = this.hiddenDim;
    const gH = this.bufGH;
    const rBuf = this.bufR;
    const zBuf = this.bufZ;
    const nBuf = this.bufN;
    const hPrevBuf = this.bufHPrev;
    const duSeq = this.bufDU;
    const dg_h = this.bufDGH;
    const dhPrev = this.bufDHPrev;
    const dhNext = this.bufDHNext;
    const wData = this.W_hh.data;
    const gradWhh = this.grad_W_hh.data;
    const gradBhh = this.grad_b_hh;
    const gradBih = this.grad_b_ih;

    dhNext.fill(0);

    for (let s = 0; s < T; s++) {
      const t = reverse ? s : (T - 1 - s);
      const hOffset = t * H;
      const uOffset = t * 3 * H;
      const dHRow = t * outD + outOffset;

      for (let j = 0; j < H; j++) {
        const delta = dH[dHRow + j] + dhNext[j];
        const r = rBuf[hOffset + j];
        const z = zBuf[hOffset + j];
        const n = nBuf[hOffset + j];
        const hp = hPrevBuf[hOffset + j];

        // 1. Candidate gradient
        const dn = delta * z;
        const dCandLinear = dn * (1.0 - n * n);
        duSeq[uOffset + 2 * H + j] = dCandLinear;
        dg_h[2 * H + j] = dCandLinear * r;
        const dr = dCandLinear * gH[uOffset + 2 * H + j];

        // 2. Update gate gradient
        const dz = delta * (n - hp);
        const dZLinear = dz * z * (1.0 - z);
        duSeq[uOffset + H + j] = dZLinear;
        dg_h[H + j] = dZLinear;

        // 3. Reset gate gradient
        const dRLinear = dr * r * (1.0 - r);
        duSeq[uOffset + j] = dRLinear;
        dg_h[j] = dRLinear;

        // 4. Direct contribution to h_prev
        dhPrev[j] = delta * (1.0 - z);
      }

      for (let i = 0; i < 3 * H; i++) {
        const g = dg_h[i];
        gradBhh[i] += g;
        gradBih[i] += duSeq[uOffset + i];

        const rowOffset = i * H;
        for (let j = 0; j < H; j++) {
          gradWhh[rowOffset + j] += g * hPrevBuf[hOffset + j];
          dhPrev[j] += wData[rowOffset + j] * g;
        }
      }

      for (let j = 0; j < H; j++) {
        dhNext[j] = dhPrev[j];
      }
    }

    this.accumulateInputGradients(duSeq, X, T);
    const dX = this.computeInputGradients(duSeq, T, this.bufDX);
    return { dX, dh0: dhNext };
  }

  /**
   * Forward pass for a single time step
   * 
   * @param {Float32Array} u_t - Pre-projected input [3*H]
   * @param {Float32Array} h_prev - Previous hidden state [H]
   * @returns {Object} { h_t, cache: { h_prev, r, z, n, g_hr, g_hz, g_hn, u_t } }
   */
  forwardStep(u_t, h_prev) {
    const H = this.hiddenDim;
    const h_t = new Float32Array(H);
    const r = new Float32Array(H);
    const z = new Float32Array(H);
    const n = new Float32Array(H);

    const g_h = new Float32Array(3 * H);
    const wData = this.W_hh.data;

    // Compute g_h = W_hh * h_prev + b_hh
    for (let i = 0; i < 3 * H; i++) {
      let sum = this.b_hh[i];
      const wRowOffset = i * H;
      for (let j = 0; j < H; j++) {
        sum += wData[wRowOffset + j] * h_prev[j];
      }
      g_h[i] = sum;
    }

    // Compute gates
    for (let j = 0; j < H; j++) {
      r[j] = sigmoid(u_t[j] + g_h[j]);
      z[j] = sigmoid(u_t[H + j] + g_h[H + j]);
      // Candidate: tanh(u_n + r * g_hn)
      const candLinear = u_t[2 * H + j] + r[j] * g_h[2 * H + j];
      n[j] = Math.tanh(candLinear);
      // New hidden state
      h_t[j] = (1.0 - z[j]) * h_prev[j] + z[j] * n[j];
    }

    return {
      h_t,
      cache: {
        h_prev,
        r,
        z,
        n,
        g_h,
        u_t,
      },
    };
  }

  /**
   * Backward pass for a single time step
   * 
   * @param {Float32Array} dh_t - Gradient w.r.t. current hidden state h_t [H]
   * @param {Object} cache - Cache from forwardStep
   * @returns {Object} { dh_prev: Float32Array [H], du_t: Float32Array [3*H] }
   */
  backwardStep(dh_t, cache) {
    const H = this.hiddenDim;
    const { h_prev, r, z, n, g_h } = cache;

    const dh_prev = new Float32Array(H);
    const du_t = new Float32Array(3 * H);
    const dg_h = new Float32Array(3 * H);

    for (let j = 0; j < H; j++) {
      const delta = dh_t[j];

      // 1. Candidate gradient
      const dn = delta * z[j];
      const dCandLinear = dn * (1.0 - n[j] * n[j]); // tanh derivative
      du_t[2 * H + j] = dCandLinear;
      dg_h[2 * H + j] = dCandLinear * r[j];
      const dr = dCandLinear * g_h[2 * H + j];

      // 2. Update gate gradient
      const dz = delta * (n[j] - h_prev[j]);
      const dZLinear = dz * z[j] * (1.0 - z[j]); // sigmoid derivative
      du_t[H + j] = dZLinear;
      dg_h[H + j] = dZLinear;

      // 3. Reset gate gradient
      const dRLinear = dr * r[j] * (1.0 - r[j]); // sigmoid derivative
      du_t[j] = dRLinear;
      dg_h[j] = dRLinear;

      // 4. Direct contribution to h_prev
      dh_prev[j] = delta * (1.0 - z[j]);
    }

    // Accumulate gradients for W_hh and b_hh: dW_hh = dg_h * h_prev^T
    const wData = this.W_hh.data;
    const gradW = this.grad_W_hh.data;

    for (let i = 0; i < 3 * H; i++) {
      const g = dg_h[i];
      this.grad_b_hh[i] += g;
      this.grad_b_ih[i] += du_t[i]; // b_ih receives du_t

      const rowOffset = i * H;
      for (let j = 0; j < H; j++) {
        gradW[rowOffset + j] += g * h_prev[j];
        // Propagate dg_h through W_hh to dh_prev
        dh_prev[j] += wData[rowOffset + j] * g;
      }
    }

    return { dh_prev, du_t };
  }

  /**
   * Accumulates W_ih gradients from the full sequence of du [T x 3*H] and X [T x inDim]
   * grad_W_ih += du^T * X
   */
  accumulateInputGradients(duSeq, X, T) {
    const H = this.hiddenDim;
    const inD = this.inDim;
    const gradW = this.grad_W_ih.data;

    for (let t = 0; t < T; t++) {
      const uOffset = t * 3 * H;
      const xOffset = t * inD;

      for (let i = 0; i < 3 * H; i++) {
        const du = duSeq[uOffset + i];
        const rowOffset = i * inD;
        for (let j = 0; j < inD; j++) {
          gradW[rowOffset + j] += du * X[xOffset + j];
        }
      }
    }
  }

  /**
   * Propagates gradients back to input sequence X [T x inDim]
   * dX = du * W_ih
   */
  computeInputGradients(duSeq, T, outDX = null) {
    const H = this.hiddenDim;
    const inD = this.inDim;
    const dX = outDX || new Float32Array(T * inD);
    if (outDX) dX.fill(0, 0, T * inD);
    const wData = this.W_ih.data;

    for (let t = 0; t < T; t++) {
      const uOffset = t * 3 * H;
      const xOffset = t * inD;

      for (let i = 0; i < 3 * H; i++) {
        const du = duSeq[uOffset + i];
        const wRowOffset = i * inD;
        for (let j = 0; j < inD; j++) {
          dX[xOffset + j] += du * wData[wRowOffset + j];
        }
      }
    }
    return dX;
  }
}

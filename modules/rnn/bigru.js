import { GruCell } from './gru-cell.js';

/**
 * GruLayer (Supports Unidirectional and Bidirectional GRU)
 * 
 * If bidirectional = true:
 *   Runs a forward GRU (t = 0 -> T-1) and a backward GRU (t = T-1 -> 0),
 *   concatenating their hidden states into a 2*H dimensional representation:
 *     H_t = [ h_fwd_t ; h_bwd_t ]
 * If bidirectional = false:
 *   Runs only the forward GRU (H_t = h_fwd_t of dimension H).
 */
export class BidirectionalGru {
  constructor(inDim, hiddenDim, bidirectional = true) {
    this.inDim = inDim;
    this.hiddenDim = hiddenDim;
    this.bidirectional = bidirectional;
    this.outDim = bidirectional ? 2 * hiddenDim : hiddenDim;

    this.cellFwd = new GruCell(inDim, hiddenDim);
    this.cellBwd = bidirectional ? new GruCell(inDim, hiddenDim) : null;

    this.cachedX = null;
    this.cachedT = 0;
    this.cachedFwd = [];
    this.cachedBwd = [];
    this.cachedU_fwd = null;
    this.cachedU_bwd = null;
  }

  zeroGrad() {
    this.cellFwd.zeroGrad();
    if (this.cellBwd) {
      this.cellBwd.zeroGrad();
    }
  }

  /**
   * Forward pass across sequence of length T
   * 
   * @param {Float32Array} X - Input features [T x inDim]
   * @param {number} T - Sequence length
   * @param {Float32Array|null} h0_fwd - Optional initial hidden state for forward GRU [H]
   * @returns {Float32Array} H - Hidden states [T x outDim]
   */
  forward(X, T, h0_fwd = null) {
    const H = this.hiddenDim;
    const outD = this.outDim;
    this.cachedX = X;
    this.cachedT = T;
    this.cachedFwd = new Array(T);
    this.cachedBwd = this.bidirectional ? new Array(T) : null;

    // 1. Parallel Pre-projection of inputs
    this.cachedU_fwd = this.cellFwd.projectInputSequence(X, T);
    if (this.bidirectional) {
      this.cachedU_bwd = this.cellBwd.projectInputSequence(X, T);
    }

    const outH = new Float32Array(T * outD);

    // 2. Forward GRU: t = 0 -> T-1
    let hPrevFwd = h0_fwd ? new Float32Array(h0_fwd) : new Float32Array(H);

    for (let t = 0; t < T; t++) {
      const uOffset = t * 3 * H;
      const u_t = this.cachedU_fwd.subarray(uOffset, uOffset + 3 * H);

      const stepResult = this.cellFwd.forwardStep(u_t, hPrevFwd);
      this.cachedFwd[t] = stepResult;
      hPrevFwd = stepResult.h_t;

      // Copy to output [t, 0..H-1]
      const outOffset = t * outD;
      outH.set(stepResult.h_t, outOffset);
    }

    // 3. Backward GRU: t = T-1 -> 0 (if bidirectional)
    if (this.bidirectional) {
      let hPrevBwd = new Float32Array(H); // Backward pass starts at 0

      for (let t = T - 1; t >= 0; t--) {
        const uOffset = t * 3 * H;
        const u_t = this.cachedU_bwd.subarray(uOffset, uOffset + 3 * H);

        const stepResult = this.cellBwd.forwardStep(u_t, hPrevBwd);
        this.cachedBwd[t] = stepResult;
        hPrevBwd = stepResult.h_t;

        // Copy to output [t, H..2*H-1]
        const outOffset = t * outD + H;
        outH.set(stepResult.h_t, outOffset);
      }
    }

    return outH;
  }

  /**
   * Backward pass (Backpropagation Through Time)
   * 
   * @param {Float32Array} dH - Gradient w.r.t. hidden states [T x outDim]
   * @returns {Object} { dX: Float32Array [T x inDim], dh0_fwd: Float32Array [H] }
   */
  backward(dH) {
    const H = this.hiddenDim;
    const outD = this.outDim;
    const T = this.cachedT;
    const X = this.cachedX;

    const du_fwd_seq = new Float32Array(T * 3 * H);

    // 1. Forward GRU BPTT: propagates backward from t = T-1 -> 0
    let dhNextFwd = new Float32Array(H);
    let dh0_fwd = null;

    for (let t = T - 1; t >= 0; t--) {
      const dh_curr = new Float32Array(H);
      const dH_offset = t * outD;

      // Direct gradient on forward hidden state at step t
      for (let j = 0; j < H; j++) {
        dh_curr[j] = dH[dH_offset + j] + dhNextFwd[j];
      }

      const { dh_prev, du_t } = this.cellFwd.backwardStep(dh_curr, this.cachedFwd[t].cache);
      du_fwd_seq.set(du_t, t * 3 * H);
      dhNextFwd = dh_prev;

      if (t === 0) {
        dh0_fwd = dh_prev;
      }
    }

    // Accumulate input weights for forward cell
    this.cellFwd.accumulateInputGradients(du_fwd_seq, X, T);
    const dX_fwd = this.cellFwd.computeInputGradients(du_fwd_seq, T);

    if (!this.bidirectional) {
      return { dX: dX_fwd, dh0_fwd };
    }

    // 2. Backward GRU BPTT: propagates forward from t = 0 -> T-1
    const du_bwd_seq = new Float32Array(T * 3 * H);
    let dhNextBwd = new Float32Array(H);

    for (let t = 0; t < T; t++) {
      const dh_curr = new Float32Array(H);
      const dH_offset = t * outD + H;

      // Direct gradient on backward hidden state at step t
      for (let j = 0; j < H; j++) {
        dh_curr[j] = dH[dH_offset + j] + dhNextBwd[j];
      }

      const { dh_prev, du_t } = this.cellBwd.backwardStep(dh_curr, this.cachedBwd[t].cache);
      du_bwd_seq.set(du_t, t * 3 * H);
      dhNextBwd = dh_prev;
    }

    // Accumulate input weights for backward cell
    this.cellBwd.accumulateInputGradients(du_bwd_seq, X, T);
    const dX_bwd = this.cellBwd.computeInputGradients(du_bwd_seq, T);

    const dX = new Float32Array(T * this.inDim);
    for (let i = 0; i < dX.length; i++) {
      dX[i] = dX_fwd[i] + dX_bwd[i];
    }

    return { dX, dh0_fwd };
  }
}

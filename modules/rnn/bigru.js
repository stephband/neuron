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
    this.capT = 0;
    this.bufOutH = null;
    this.bufDX = null;
  }

  ensureCapacity(T) {
    if (T <= this.capT) return;
    this.capT = T;
    this.bufOutH = new Float32Array(T * this.outDim);
    this.bufDX = new Float32Array(T * this.inDim);
  }

  zeroGrad() {
    this.cellFwd.zeroGrad();
    if (this.cellBwd) {
      this.cellBwd.zeroGrad();
    }
  }

  getParameters(prefix = "") {
    const p = prefix ? `${prefix}_` : "";
    const params = [
      {
        name: `${p}Wih_fwd`,
        weight: this.cellFwd.W_ih.data,
        grad: this.cellFwd.grad_W_ih.data,
      },
      {
        name: `${p}Whh_fwd`,
        weight: this.cellFwd.W_hh.data,
        grad: this.cellFwd.grad_W_hh.data,
      },
      {
        name: `${p}bih_fwd`,
        weight: this.cellFwd.b_ih,
        grad: this.cellFwd.grad_b_ih,
      },
      {
        name: `${p}bhh_fwd`,
        weight: this.cellFwd.b_hh,
        grad: this.cellFwd.grad_b_hh,
      },
    ];
    if (this.bidirectional && this.cellBwd) {
      params.push(
        {
          name: `${p}Wih_bwd`,
          weight: this.cellBwd.W_ih.data,
          grad: this.cellBwd.grad_W_ih.data,
        },
        {
          name: `${p}Whh_bwd`,
          weight: this.cellBwd.W_hh.data,
          grad: this.cellBwd.grad_W_hh.data,
        },
        {
          name: `${p}bih_bwd`,
          weight: this.cellBwd.b_ih,
          grad: this.cellBwd.grad_b_ih,
        },
        {
          name: `${p}bhh_bwd`,
          weight: this.cellBwd.b_hh,
          grad: this.cellBwd.grad_b_hh,
        },
      );
    }
    return params;
  }

  get parameters() {
    return this.getParameters();
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
    this.ensureCapacity(T);

    const outH = this.bufOutH.subarray(0, T * outD);

    // 1. Forward GRU: t = 0 -> T-1 (writes columns 0..H-1)
    this.cellFwd.forwardSequence(X, T, h0_fwd, outH, outD, 0, false);

    // 2. Backward GRU: t = T-1 -> 0 (writes columns H..2*H-1)
    if (this.bidirectional) {
      this.cellBwd.forwardSequence(X, T, null, outH, outD, H, true);
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
    const lenX = T * this.inDim;

    // 1. Forward GRU BPTT: propagates backward from t = T-1 -> 0
    const { dX: dX_fwd, dh0: dh0_fwd } = this.cellFwd.backwardSequence(
      dH,
      X,
      T,
      outD,
      0,
      false,
    );

    if (!this.bidirectional) {
      return { dX: dX_fwd.subarray(0, lenX), dh0_fwd };
    }

    // 2. Backward GRU BPTT: propagates forward from t = 0 -> T-1
    const { dX: dX_bwd } = this.cellBwd.backwardSequence(
      dH,
      X,
      T,
      outD,
      H,
      true,
    );

    const dX = this.bufDX.subarray(0, lenX);
    for (let i = 0; i < lenX; i++) {
      dX[i] = dX_fwd[i] + dX_bwd[i];
    }

    return { dX, dh0_fwd };
  }
}

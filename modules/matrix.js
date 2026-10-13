/**
 * Matrix2D: Contiguous Float32Array 2D Matrix
 * 
 * Stored in row-major order: index = row * cols + col.
 * This directly mirrors the memory layout of WebGPU storage buffers.
 */
export class Matrix2D {
  constructor(rows, cols, data = null) {
    this.rows = rows;
    this.cols = cols;
    if (data) {
      if (data.length !== rows * cols) {
        throw new Error(`Data length (${data.length}) does not match shape (${rows}x${cols})`);
      }
      this.data = data instanceof Float32Array ? data : new Float32Array(data);
    } else {
      this.data = new Float32Array(rows * cols);
    }
  }

  static zeros(rows, cols) {
    return new Matrix2D(rows, cols);
  }

  /**
   * He (Kaiming) Normal Initialization
   * Ideal for ReLU activations: stddev = sqrt(2 / fan_in)
   */
  static randomHe(rows, cols) {
    const mat = new Matrix2D(rows, cols);
    const std = Math.sqrt(2.0 / rows);
    for (let i = 0; i < mat.data.length; i++) {
      // Box-Muller transform for standard normal distribution
      const u1 = Math.max(1e-7, Math.random());
      const u2 = Math.random();
      const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      mat.data[i] = z0 * std;
    }
    return mat;
  }

  /**
   * Xavier (Glorot) Uniform Initialization
   * Useful for output layers or sigmoid/tanh
   */
  static randomXavier(rows, cols) {
    const mat = new Matrix2D(rows, cols);
    const limit = Math.sqrt(6.0 / (rows + cols));
    for (let i = 0; i < mat.data.length; i++) {
      mat.data[i] = (Math.random() * 2 - 1) * limit;
    }
    return mat;
  }

  get(r, c) {
    return this.data[r * this.cols + c];
  }

  set(r, c, val) {
    this.data[r * this.cols + c] = val;
  }

  clone() {
    const copy = new Matrix2D(this.rows, this.cols);
    copy.data.set(this.data);
    return copy;
  }

  fill(val) {
    this.data.fill(val);
    return this;
  }

  /**
   * Matrix Multiplication: C = A * B
   * A: [M x K], B: [K x N] -> C: [M x N]
   */
  static matmul(A, B, out = null) {
    if (A.cols !== B.rows) {
      throw new Error(`Dimension mismatch: matmul(${A.rows}x${A.cols}, ${B.rows}x${B.cols})`);
    }
    const M = A.rows;
    const K = A.cols;
    const N = B.cols;
    const C = out || new Matrix2D(M, N);

    const aData = A.data;
    const bData = B.data;
    const cData = C.data;

    // Cache-friendly row-major order
    for (let m = 0; m < M; m++) {
      const mOffset = m * K;
      const cOffset = m * N;
      for (let k = 0; k < K; k++) {
        const aVal = aData[mOffset + k];
        const bOffset = k * N;
        for (let n = 0; n < N; n++) {
          cData[cOffset + n] += aVal * bData[bOffset + n];
        }
      }
    }
    return C;
  }

  /**
   * Transpose: B = A^T
   * A: [rows x cols] -> B: [cols x rows]
   */
  static transpose(A, out = null) {
    const M = A.rows;
    const N = A.cols;
    const B = out || new Matrix2D(N, M);
    const aData = A.data;
    const bData = B.data;

    for (let r = 0; r < M; r++) {
      for (let c = 0; c < N; c++) {
        bData[c * M + r] = aData[r * N + c];
      }
    }
    return B;
  }

  /**
   * Add Bias Vector: Z = X + b (broadcast over batch rows)
   * X: [batch x dim], b: [1 x dim]
   */
  static addBias(X, b, out = null) {
    if (b.cols !== X.cols || b.rows !== 1) {
      throw new Error(`Bias shape (${b.rows}x${b.cols}) must match (1x${X.cols})`);
    }
    const result = out || new Matrix2D(X.rows, X.cols);
    const xData = X.data;
    const bData = b.data;
    const rData = result.data;
    const cols = X.cols;

    for (let r = 0; r < X.rows; r++) {
      const rowOffset = r * cols;
      for (let c = 0; c < cols; c++) {
        rData[rowOffset + c] = xData[rowOffset + c] + bData[c];
      }
    }
    return result;
  }

  /**
   * Sum across batch rows (for bias gradient): db = sum_rows(dZ)
   * dZ: [batch x dim] -> db: [1 x dim]
   */
  static sumRows(dZ, out = null) {
    const result = out || new Matrix2D(1, dZ.cols);
    const rData = result.data;
    const zData = dZ.data;
    const cols = dZ.cols;
    rData.fill(0);

    for (let r = 0; r < dZ.rows; r++) {
      const rowOffset = r * cols;
      for (let c = 0; c < cols; c++) {
        rData[c] += zData[rowOffset + c];
      }
    }
    return result;
  }

  /**
   * Element-wise multiplication: C = A (hadamard) B
   */
  static elementWiseMul(A, B, out = null) {
    const C = out || new Matrix2D(A.rows, A.cols);
    for (let i = 0; i < A.data.length; i++) {
      C.data[i] = A.data[i] * B.data[i];
    }
    return C;
  }
}

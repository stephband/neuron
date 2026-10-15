import { Matrix2D } from "../matrix.js";

/**
 * Numerically stable Softmax for a batch of rows [T x C]
 */
function softmax(logits, T, C) {
    const probs = new Float32Array(T * C);
    for (let t = 0; t < T; t++) {
        const rowOffset = t * C;
        let maxVal = -Infinity;
        for (let c = 0; c < C; c++) {
            const v = logits[rowOffset + c];
            if (v > maxVal) maxVal = v;
        }
        let sumExp = 0.0;
        for (let c = 0; c < C; c++) {
            const expVal = Math.exp(logits[rowOffset + c] - maxVal);
            probs[rowOffset + c] = expVal;
            sumExp += expVal;
        }
        const invSum = 1.0 / (sumExp + 1e-9);
        for (let c = 0; c < C; c++) {
            probs[rowOffset + c] *= invSum;
        }
    }
    return probs;
}

/**
 * LinearHead: Composable output projection head for recurrent networks
 */
export default class LinearHead {
    static of(options) {
        return new LinearHead(options);
    }

    constructor({
        name = "output",
        layerIndex = -1,
        inDim,
        outDim,
        lossWeight = 1.0,
        priorBias = null,
    }) {
        this.name = name;
        this.layerIndex = layerIndex;
        this.inDim = inDim;
        this.outDim = outDim;
        this.lossWeight = lossWeight;
        this.priorBias = priorBias ? new Float32Array(priorBias) : null;

        this.W = Matrix2D.randomXavier(outDim, inDim);
        this.b = new Float32Array(outDim);
        this.gradW = Matrix2D.zeros(outDim, inDim);
        this.gradB = new Float32Array(outDim);
    }

    forward(H, T) {
        const logits = new Float32Array(T * this.outDim);
        const H_data = H.data || H;
        const W_data = this.W.data;
        const b_data = this.b;
        const prior = this.priorBias;
        const outD = this.outDim;
        const inD = this.inDim;

        for (let t = 0; t < T; t++) {
            const hOffset = t * inD;
            const lOffset = t * outD;
            for (let c = 0; c < outD; c++) {
                let sum = b_data[c] + (prior ? prior[c] : 0.0);
                const wOffset = c * inD;
                for (let j = 0; j < inD; j++) {
                    sum += H_data[hOffset + j] * W_data[wOffset + j];
                }
                logits[lOffset + c] = sum;
            }
        }

        const probs = softmax(logits, T, outD);
        const preds = new Uint32Array(T);
        for (let t = 0; t < T; t++) {
            const lOffset = t * outD;
            let maxIdx = 0;
            let maxVal = -Infinity;
            for (let c = 0; c < outD; c++) {
                const v = logits[lOffset + c];
                if (v > maxVal) {
                    maxVal = v;
                    maxIdx = c;
                }
            }
            preds[t] = maxIdx;
        }

        return { logits, probs, preds };
    }

    backward(gradLogits, H, T, dH) {
        const H_data = H.data || H;
        const W_data = this.W.data;
        const gradW_data = this.gradW.data;
        const gradB_data = this.gradB;
        const dH_data = dH.data || dH;
        const outD = this.outDim;
        const inD = this.inDim;

        // 1. gradW += gradLogits^T * H; gradB += sum(gradLogits)
        for (let t = 0; t < T; t++) {
            const lOffset = t * outD;
            const hOffset = t * inD;
            for (let c = 0; c < outD; c++) {
                const dL = gradLogits[lOffset + c];
                gradB_data[c] += dL;
                const wOffset = c * inD;
                for (let j = 0; j < inD; j++) {
                    gradW_data[wOffset + j] += dL * H_data[hOffset + j];
                }
            }
        }

        // 2. dH += gradLogits * W
        for (let t = 0; t < T; t++) {
            const lOffset = t * outD;
            const hOffset = t * inD;
            for (let c = 0; c < outD; c++) {
                const dL = gradLogits[lOffset + c];
                const wOffset = c * inD;
                for (let j = 0; j < inD; j++) {
                    dH_data[hOffset + j] += dL * W_data[wOffset + j];
                }
            }
        }
    }

    zeroGrad() {
        this.gradW.fill(0);
        this.gradB.fill(0);
    }

    getParameters() {
        return [
            {
                name: `head_${this.name}_W`,
                weight: this.W.data,
                grad: this.gradW.data,
            },
            { name: `head_${this.name}_b`, weight: this.b, grad: this.gradB },
        ];
    }
}

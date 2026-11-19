import { normalizeWeights } from "../deserialize.js";
import {
    packConcatCpu,
    resolveHeadMeta,
    resolveLayerConcat,
} from "./concat.js";

function sigmoid(x) {
    return 1.0 / (1.0 + Math.exp(-Math.max(-85, Math.min(85, x))));
}

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

function runDirection(
    X,
    T,
    inD,
    H,
    outH,
    outD,
    outOffset,
    Wih,
    Whh,
    bih,
    bhh,
    h0,
    reverse,
) {
    const H2 = 2 * H;
    const H3 = 3 * H;
    const hPrev = new Float32Array(H);
    const u = new Float32Array(H3);
    const gh = new Float32Array(H3);

    if (h0 && !reverse) {
        hPrev.set(h0);
    }

    for (let s = 0; s < T; s++) {
        const t = reverse ? (T - 1 - s) : s;
        const xOff = t * inD;
        const outRow = t * outD + outOffset;

        for (let i = 0; i < H3; i++) {
            let s1 = bih[i];
            let s2 = bhh[i];
            const w1 = i * inD;
            const w2 = i * H;
            for (let j = 0; j < inD; j++) s1 += Wih[w1 + j] * X[xOff + j];
            for (let j = 0; j < H; j++) s2 += Whh[w2 + j] * hPrev[j];
            u[i] = s1;
            gh[i] = s2;
        }

        for (let j = 0; j < H; j++) {
            const r = sigmoid(u[j] + gh[j]);
            const z = sigmoid(u[H + j] + gh[H + j]);
            const n = Math.tanh(u[H2 + j] + r * gh[H2 + j]);
            const hNew = (1.0 - z) * hPrev[j] + z * n;
            hPrev[j] = hNew;
            outH[outRow + j] = hNew;
        }
    }
}

function projectHead(head, H, T) {
    const outD = head.outDim;
    const inD = head.inDim;
    const W = head.W;
    const b = head.b;
    const prior = head.priorBias;
    const logits = new Float32Array(T * outD);

    for (let t = 0; t < T; t++) {
        const hOff = t * inD;
        const lOff = t * outD;
        for (let c = 0; c < outD; c++) {
            let sum = b[c] + (prior ? prior[c] : 0.0);
            const wOff = c * inD;
            for (let j = 0; j < inD; j++) {
                sum += H[hOff + j] * W[wOff + j];
            }
            logits[lOff + c] = sum;
        }
    }

    const probs = softmax(logits, T, outD);
    const preds = new Uint32Array(T);
    for (let t = 0; t < T; t++) {
        const lOff = t * outD;
        let maxIdx = 0;
        let maxVal = -Infinity;
        for (let c = 0; c < outD; c++) {
            const v = logits[lOff + c];
            if (v > maxVal) {
                maxVal = v;
                maxIdx = c;
            }
        }
        preds[t] = maxIdx;
    }

    return { logits, probs, preds };
}

/**
 * RNN: Lightweight CPU inference engine for recurrent architectures
 * (for training / BPTT use `network-train.js`).
 */
export default class RNN {
    static async from(definition, weights) {
        return new this(definition, weights);
    }

    constructor({
        inDim,
        layers = [{ hiddenDim: 48, bidirectional: true }],
        priming = null,
        heads = [{
            name: "output",
            layerIndex: -1,
            outDim: 1,
            priorBias: null,
        }],
    }, weights = null) {
        if (!inDim || inDim <= 0) {
            throw new Error(`Invalid inDim: ${inDim}`);
        }

        this.inDim = inDim;
        const headMetaByName = resolveHeadMeta(heads, layers.length);
        this.parameters = [];

        // 1. Layers
        this.layers = [];
        for (let l = 0; l < layers.length; l++) {
            const item = layers[l];
            const baseInDim = l === 0 ? inDim : this.layers[l - 1].outDim;
            const concatSources = resolveLayerConcat(
                item.concat,
                l,
                inDim,
                headMetaByName,
            );
            const layerInDim = baseInDim +
                concatSources.reduce((sum, src) => sum + src.dim, 0);
            const hiddenDim = item.hiddenDim || 48;
            const bidirectional = item.bidirectional !== undefined
                ? item.bidirectional
                : true;
            const outDim = bidirectional ? 2 * hiddenDim : hiddenDim;
            const H3 = 3 * hiddenDim;

            const layer = {
                inDim: layerInDim,
                hiddenDim,
                bidirectional,
                outDim,
                concatSources,
                Wih_fwd: new Float32Array(H3 * layerInDim),
                Whh_fwd: new Float32Array(H3 * hiddenDim),
                bih_fwd: new Float32Array(H3),
                bhh_fwd: new Float32Array(H3),
                Wih_bwd: bidirectional ? new Float32Array(H3 * layerInDim) : null,
                Whh_bwd: bidirectional ? new Float32Array(H3 * hiddenDim) : null,
                bih_bwd: bidirectional ? new Float32Array(H3) : null,
                bhh_bwd: bidirectional ? new Float32Array(H3) : null,
            };
            this.layers.push(layer);
        }

        // 2. Priming (Optional)
        this.priming = null;
        if (priming) {
            const H0 = this.layers[0].hiddenDim;
            const pDim = priming.inDim;
            this.priming = {
                type: priming.type || "categorical",
                inDim: pDim,
                W: new Float32Array(H0 * pDim),
                b: new Float32Array(H0),
            };
            this.parameters.push(
                { name: "priming_W", weight: this.priming.W },
                { name: "priming_b", weight: this.priming.b },
            );
        }

        for (let l = 0; l < this.layers.length; l++) {
            const layer = this.layers[l];
            this.parameters.push(
                { name: `layer${l}_Wih_fwd`, weight: layer.Wih_fwd },
                { name: `layer${l}_Whh_fwd`, weight: layer.Whh_fwd },
                { name: `layer${l}_bih_fwd`, weight: layer.bih_fwd },
                { name: `layer${l}_bhh_fwd`, weight: layer.bhh_fwd },
            );
            if (layer.bidirectional) {
                this.parameters.push(
                    { name: `layer${l}_Wih_bwd`, weight: layer.Wih_bwd },
                    { name: `layer${l}_Whh_bwd`, weight: layer.Whh_bwd },
                    { name: `layer${l}_bih_bwd`, weight: layer.bih_bwd },
                    { name: `layer${l}_bhh_bwd`, weight: layer.bhh_bwd },
                );
            }
        }

        // 3. Output Heads
        this.heads = [];
        for (let h = 0; h < heads.length; h++) {
            const item = heads[h];
            let layerIdx = item.layerIndex !== undefined ? item.layerIndex : -1;
            if (layerIdx < 0) layerIdx = this.layers.length + layerIdx;
            if (layerIdx < 0 || layerIdx >= this.layers.length) {
                throw new Error(
                    `Head layerIndex ${layerIdx} out of range [0, ${
                        this.layers.length - 1
                    }]`,
                );
            }
            const headInDim = this.layers[layerIdx].outDim;
            const head = {
                name: item.name || `head_${h}`,
                layerIndex: layerIdx,
                inDim: headInDim,
                outDim: item.outDim,
                priorBias: item.priorBias ? new Float32Array(item.priorBias) : null,
                W: new Float32Array(item.outDim * headInDim),
                b: new Float32Array(item.outDim),
            };
            this.heads.push(head);
            this.parameters.push(
                { name: `head_${head.name}_W`, weight: head.W },
                { name: `head_${head.name}_b`, weight: head.b },
            );
        }

        if (weights) {
            this.loadWeights(weights);
        }
    }

    get parameterCount() {
        let total = 0;
        for (let i = 0; i < this.parameters.length; i++) {
            total += this.parameters[i].weight.length;
        }
        return total;
    }

    loadWeights(weightsOrArray) {
        if (
            weightsOrArray instanceof ArrayBuffer ||
            ArrayBuffer.isView(weightsOrArray)
        ) {
            const flat = normalizeWeights(weightsOrArray, this.parameterCount);
            let offset = 0;
            for (const p of this.parameters) {
                p.weight.set(flat.subarray(offset, offset + p.weight.length));
                offset += p.weight.length;
            }
            return this;
        }
        const weights = weightsOrArray.weights || weightsOrArray;
        for (const p of this.parameters) {
            if (weights[p.name]) {
                p.weight.set(weights[p.name]);
            }
        }
        return this;
    }

    computePrimingH0(condition) {
        if (!this.priming) return null;
        const H0 = this.layers[0].hiddenDim;
        const h0 = new Float32Array(H0);
        const b = this.priming.b;
        const W = this.priming.W;
        const pDim = this.priming.inDim;

        if (this.priming.type === "categorical") {
            const col = Number(condition);
            if (col >= 0 && col < pDim) {
                for (let i = 0; i < H0; i++) {
                    h0[i] = Math.tanh(W[i * pDim + col] + b[i]);
                }
            } else {
                for (let i = 0; i < H0; i++) h0[i] = Math.tanh(b[i]);
            }
        } else {
            const condVec = condition;
            for (let i = 0; i < H0; i++) {
                let sum = b[i];
                const rowOffset = i * pDim;
                for (let j = 0; j < pDim; j++) {
                    sum += W[rowOffset + j] * condVec[j];
                }
                h0[i] = Math.tanh(sum);
            }
        }
        return h0;
    }

    forward(X, T, options = {}) {
        if (T <= 0) throw new Error("Sequence length T must be > 0");

        const rawInput = X instanceof Float32Array
            ? X
            : (X.data || new Float32Array(X));

        let h0 = options.h0 || null;
        if (!h0 && options.condition !== undefined && this.priming) {
            h0 = this.computePrimingH0(options.condition);
        }

        const layerActivations = new Array(this.layers.length);
        const headOutputs = {};

        for (let l = 0; l < this.layers.length; l++) {
            const layer = this.layers[l];
            const layerH0 = l === 0 ? h0 : null;
            const baseInput = l === 0 ? rawInput : layerActivations[l - 1];
            const baseDim = l === 0 ? this.inDim : this.layers[l - 1].outDim;
            const layerInput = layer.concatSources.length
                ? packConcatCpu(
                    baseInput,
                    baseDim,
                    layer.concatSources,
                    rawInput,
                    headOutputs,
                    T,
                    layer.inDim,
                )
                : baseInput;

            const H = new Float32Array(T * layer.outDim);
            runDirection(
                layerInput,
                T,
                layer.inDim,
                layer.hiddenDim,
                H,
                layer.outDim,
                0,
                layer.Wih_fwd,
                layer.Whh_fwd,
                layer.bih_fwd,
                layer.bhh_fwd,
                layerH0,
                false,
            );
            if (layer.bidirectional) {
                runDirection(
                    layerInput,
                    T,
                    layer.inDim,
                    layer.hiddenDim,
                    H,
                    layer.outDim,
                    layer.hiddenDim,
                    layer.Wih_bwd,
                    layer.Whh_bwd,
                    layer.bih_bwd,
                    layer.bhh_bwd,
                    null,
                    true,
                );
            }

            layerActivations[l] = H;

            for (const head of this.heads) {
                if (head.layerIndex === l) {
                    headOutputs[head.name] = projectHead(head, H, T);
                }
            }
        }

        return headOutputs;
    }

    async predict(X, T, options = {}) {
        return this.forward(X, T, options);
    }
}

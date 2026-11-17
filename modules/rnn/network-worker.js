import compileWorker from 'fn/compile-worker.js';
import { normalizeWeights } from '../deserialize.js';


const workerCode = `
let layers = [];
let heads  = [];
let inDim  = 0;

function sigmoid(x) {
    return 1.0 / (1.0 + Math.exp(-Math.max(-85, Math.min(85, x))));
}

function runDirection(X, T, inD, H, outH, outD, outOffset, Wih, Whh, bih, bhh, reverse) {
    const H2    = 2 * H;
    const H3    = 3 * H;
    const hPrev = new Float32Array(H);
    const u     = new Float32Array(H3);
    const gh    = new Float32Array(H3);

    for (let s = 0; s < T; s++) {
        const t      = reverse ? (T - 1 - s) : s;
        const xOff   = t * inD;
        const outRow = t * outD + outOffset;

        for (let i = 0; i < H3; i++) {
            let s1 = bih[i];
            let s2 = bhh[i];
            const w1 = i * inD;
            const w2 = i * H;
            for (let j = 0; j < inD; j++) s1 += Wih[w1 + j] * X[xOff + j];
            for (let j = 0; j < H; j++)   s2 += Whh[w2 + j] * hPrev[j];
            u[i]  = s1;
            gh[i] = s2;
        }

        for (let j = 0; j < H; j++) {
            const r    = sigmoid(u[j] + gh[j]);
            const z    = sigmoid(u[H + j] + gh[H + j]);
            const n    = Math.tanh(u[H2 + j] + r * gh[H2 + j]);
            const hNew = (1.0 - z) * hPrev[j] + z * n;
            hPrev[j]         = hNew;
            outH[outRow + j] = hNew;
        }
    }
}

self.onmessage = function(e) {
    const msg = e.data;

    if (msg.type === 'init') {
        inDim = msg.inDim;
        const w = new Float32Array(msg.weights);
        let offset = 0;
        let layerInDim = inDim;

        layers = msg.layers.map((cfg) => {
            const H      = cfg.hiddenDim;
            const H3     = 3 * H;
            const bidi   = cfg.bidirectional !== false;
            const outDim = bidi ? 2 * H : H;
            const inD    = layerInDim;
            layerInDim   = outDim;

            const Wih_f = w.subarray(offset, offset += H3 * inD);
            const Whh_f = w.subarray(offset, offset += H3 * H);
            const bih_f = w.subarray(offset, offset += H3);
            const bhh_f = w.subarray(offset, offset += H3);

            let Wih_b = null, Whh_b = null, bih_b = null, bhh_b = null;
            if (bidi) {
                Wih_b = w.subarray(offset, offset += H3 * inD);
                Whh_b = w.subarray(offset, offset += H3 * H);
                bih_b = w.subarray(offset, offset += H3);
                bhh_b = w.subarray(offset, offset += H3);
            }

            return { inDim: inD, H, bidi, outDim, Wih_f, Whh_f, bih_f, bhh_f, Wih_b, Whh_b, bih_b, bhh_b };
        });

        heads = msg.heads.map((hCfg, h) => {
            let layerIndex = hCfg.layerIndex !== undefined ? hCfg.layerIndex : -1;
            if (layerIndex < 0) layerIndex = layers.length + layerIndex;
            const headInDim = layers[layerIndex].outDim;
            const outDim    = hCfg.outDim;

            const W = w.subarray(offset, offset += outDim * headInDim);
            const b = new Float32Array(w.subarray(offset, offset += outDim));
            if (hCfg.priorBias) {
                for (let c = 0; c < outDim; c++) b[c] += hCfg.priorBias[c];
            }

            return { name: hCfg.name || ('head_' + h), layerIndex, inDim: headInDim, outDim, W, b };
        });

        return;
    }

    const { id, buffer, T } = msg;
    const X = new Float32Array(buffer);
    const layerOutputs = new Array(layers.length);

    for (let l = 0; l < layers.length; l++) {
        const layer  = layers[l];
        const input  = l === 0 ? X : layerOutputs[l - 1];
        const outH   = new Float32Array(T * layer.outDim);

        runDirection(input, T, layer.inDim, layer.H, outH, layer.outDim, 0, layer.Wih_f, layer.Whh_f, layer.bih_f, layer.bhh_f, false);
        if (layer.bidi) {
            runDirection(input, T, layer.inDim, layer.H, outH, layer.outDim, layer.H, layer.Wih_b, layer.Whh_b, layer.bih_b, layer.bhh_b, true);
        }

        layerOutputs[l] = outH;
    }

    const outputs  = {};
    const transfer = [];

    for (let h = 0; h < heads.length; h++) {
        const head   = heads[h];
        const H_data = layerOutputs[head.layerIndex];
        const inD    = head.inDim;
        const outD   = head.outDim;
        const W      = head.W;
        const b      = head.b;
        const preds  = new Uint32Array(T);

        for (let t = 0; t < T; t++) {
            const hOff = t * inD;
            let bestIdx = 0;
            let maxVal  = -Infinity;

            for (let c = 0; c < outD; c++) {
                let sum = b[c];
                const wOff = c * inD;
                for (let j = 0; j < inD; j++) {
                    sum += H_data[hOff + j] * W[wOff + j];
                }
                if (sum > maxVal) {
                    maxVal  = sum;
                    bestIdx = c;
                }
            }

            preds[t] = bestIdx;
        }

        outputs[head.name] = { preds };
        transfer.push(preds.buffer);
    }

    self.postMessage({ id, outputs }, transfer);
};
`;


function countParameters(schema) {
    let total = 0;
    const layerOutDims = [];

    for (let l = 0; l < schema.layers.length; l++) {
        const cfg   = schema.layers[l];
        const inD   = l === 0 ? schema.inDim : layerOutDims[l - 1];
        const H     = cfg.hiddenDim;
        const bidi  = cfg.bidirectional !== false;
        const dirs  = bidi ? 2 : 1;
        total += dirs * (3 * H * inD + 3 * H * H + 6 * H);
        layerOutDims.push(bidi ? 2 * H : H);
    }

    for (let h = 0; h < schema.heads.length; h++) {
        const hCfg = schema.heads[h];
        let lIdx = hCfg.layerIndex !== undefined ? hCfg.layerIndex : -1;
        if (lIdx < 0) lIdx = schema.layers.length + lIdx;
        const inD = layerOutDims[lIdx];
        total += hCfg.outDim * inD + hCfg.outDim;
    }

    return total;
}


/**
WorkerRNN(schema, weights)

Runs a multi-layer, multi-head BiGRU recurrent network inside a compiled
Web Worker off the main thread, using zero-copy ArrayBuffer transfers.
**/

export default class WorkerRNN {
    static from(schema, weights) {
        return Promise.resolve(new WorkerRNN(schema, weights));
    }

    constructor(schema, weights) {
        this.schema  = schema;
        this.worker  = compileWorker(workerCode);
        this.nextId  = 0;
        this.pending = new Map();

        this.worker.onmessage = (e) => {
            const { id, outputs } = e.data;
            const resolve = this.pending.get(id);
            if (resolve) {
                this.pending.delete(id);
                resolve(outputs);
            }
        };

        if (weights) {
            this.loadWeights(weights);
        }
    }

    loadWeights(weights) {
        const expected = countParameters(this.schema);
        const flat     = new Float32Array(normalizeWeights(weights, expected));
        this.worker.postMessage({
            type:    'init',
            inDim:   this.schema.inDim,
            layers:  this.schema.layers.map(({ hiddenDim, bidirectional }) => ({ hiddenDim, bidirectional })),
            heads:   this.schema.heads.map(({ name, layerIndex, outDim, priorBias }) => ({ name, layerIndex, outDim, priorBias })),
            weights: flat.buffer
        }, [flat.buffer]);
    }

    predict(X, T) {
        const id     = ++this.nextId;
        const input  = X instanceof Float32Array ? X : new Float32Array(X);
        const buffer = (input.byteOffset === 0 && input.byteLength === input.buffer.byteLength)
            ? input.buffer
            : input.slice().buffer;

        return new Promise((resolve) => {
            this.pending.set(id, resolve);
            this.worker.postMessage({ id, buffer, T }, [buffer]);
        });
    }

    terminate() {
        this.worker.terminate();
    }
}

export { WorkerRNN };

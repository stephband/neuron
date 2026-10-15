import { Matrix2D } from "../matrix.js";
import { BidirectionalGru } from "./bigru.js";
import LinearHead from "./linear-head.js";

/**
 * RecurrentNetwork: Composable, Configurable Sequence-to-Sequence Recurrent Engine
 */
export default class RecurrentNetwork {
    constructor({
        inDim,
        layers = [{ hiddenDim: 48, bidirectional: true }],
        priming = null,
        heads = [{
            name: "output",
            layerIndex: -1,
            outDim: 1,
            lossWeight: 1.0,
            priorBias: null,
        }],
    }, weights = null) {
        if (!inDim || inDim <= 0) {
            throw new Error(`Invalid inDim: ${inDim}`);
        }
        this.inDim = inDim;
        this.layerConfigs = layers;
        this.primingConfig = priming;
        this.headConfigs = heads;

        // 1. Compose Recurrent Layers (accepts pre-instantiated layers or layer configs)
        this.layers = [];
        let currentInDim = inDim;

        for (let l = 0; l < layers.length; l++) {
            const item = layers[l];
            if (
                item instanceof BidirectionalGru ||
                (item.forward && item.backward)
            ) {
                this.layers.push(item);
                currentInDim = item.outDim;
            } else {
                const hiddenDim = item.hiddenDim || 48;
                const bidirectional = item.bidirectional !== undefined
                    ? item.bidirectional
                    : true;
                const layer = new BidirectionalGru(
                    currentInDim,
                    hiddenDim,
                    bidirectional,
                );
                this.layers.push(layer);
                currentInDim = layer.outDim;
            }
        }

        // 2. Condition Priming Projection (Optional)
        this.priming = null;
        if (priming) {
            const H0 = this.layers[0].hiddenDim;
            const pDim = priming.inDim;
            const pType = priming.type || "categorical";

            this.priming = {
                type: pType,
                inDim: pDim,
                W: Matrix2D.randomXavier(H0, pDim),
                b: new Float32Array(H0),
                gradW: Matrix2D.zeros(H0, pDim),
                gradB: new Float32Array(H0),
            };
        }

        // 3. Compose Output Projection Heads (accepts LinearHead instances or configs)
        this.heads = [];
        for (let h = 0; h < heads.length; h++) {
            const item = heads[h];
            if (item instanceof LinearHead) {
                this.heads.push(item);
            } else {
                let layerIdx = item.layerIndex !== undefined
                    ? item.layerIndex
                    : -1;
                if (layerIdx < 0) layerIdx = this.layers.length + layerIdx;
                if (layerIdx < 0 || layerIdx >= this.layers.length) {
                    throw new Error(
                        `Head layerIndex ${layerIdx} out of range [0, ${
                            this.layers.length - 1
                        }]`,
                    );
                }
                const layerOutDim = this.layers[layerIdx].outDim;
                const head = new LinearHead({
                    name: item.name || `head_${h}`,
                    layerIndex: layerIdx,
                    inDim: layerOutDim,
                    outDim: item.outDim,
                    lossWeight: item.lossWeight !== undefined
                        ? item.lossWeight
                        : 1.0,
                    priorBias: item.priorBias,
                });
                this.heads.push(head);
            }
        }

        // 4. Register all parameters for Optimizer
        this.stepCount = 0;
        this.parameters = [];

        if (this.priming) {
            this.parameters.push(
                {
                    name: "priming_W",
                    weight: this.priming.W.data,
                    grad: this.priming.gradW.data,
                },
                {
                    name: "priming_b",
                    weight: this.priming.b,
                    grad: this.priming.gradB,
                },
            );
        }

        for (let l = 0; l < this.layers.length; l++) {
            const layer = this.layers[l];
            this.parameters.push(
                {
                    name: `layer${l}_Wih_fwd`,
                    weight: layer.cellFwd.W_ih.data,
                    grad: layer.cellFwd.grad_W_ih.data,
                },
                {
                    name: `layer${l}_Whh_fwd`,
                    weight: layer.cellFwd.W_hh.data,
                    grad: layer.cellFwd.grad_W_hh.data,
                },
                {
                    name: `layer${l}_bih_fwd`,
                    weight: layer.cellFwd.b_ih,
                    grad: layer.cellFwd.grad_b_ih,
                },
                {
                    name: `layer${l}_bhh_fwd`,
                    weight: layer.cellFwd.b_hh,
                    grad: layer.cellFwd.grad_b_hh,
                },
            );
            if (layer.bidirectional && layer.cellBwd) {
                this.parameters.push(
                    {
                        name: `layer${l}_Wih_bwd`,
                        weight: layer.cellBwd.W_ih.data,
                        grad: layer.cellBwd.grad_W_ih.data,
                    },
                    {
                        name: `layer${l}_Whh_bwd`,
                        weight: layer.cellBwd.W_hh.data,
                        grad: layer.cellBwd.grad_W_hh.data,
                    },
                    {
                        name: `layer${l}_bih_bwd`,
                        weight: layer.cellBwd.b_ih,
                        grad: layer.cellBwd.grad_b_ih,
                    },
                    {
                        name: `layer${l}_bhh_bwd`,
                        weight: layer.cellBwd.b_hh,
                        grad: layer.cellBwd.grad_b_hh,
                    },
                );
            }
        }

        for (const head of this.heads) {
            this.parameters.push(...head.getParameters());
        }

        // Adam moment state caches
        this.adamM = new Float32Array(
            this.parameters.reduce((sum, p) => sum + p.weight.length, 0),
        );
        this.adamV = new Float32Array(this.adamM.length);
        let pOffset = 0;
        for (const p of this.parameters) {
            p.mSlice = this.adamM.subarray(pOffset, pOffset + p.weight.length);
            p.vSlice = this.adamV.subarray(pOffset, pOffset + p.weight.length);
            pOffset += p.weight.length;
        }

        this.forwardCache = null;

        if (weights) {
            this.loadWeights(weights);
        }
    }

    computePrimingH0(condition) {
        if (!this.priming) return null;
        const H0 = this.layers[0].hiddenDim;
        const h0 = new Float32Array(H0);
        const b = this.priming.b;
        const W = this.priming.W.data;
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

        const layerActivations = [];
        let currentInput = rawInput;

        for (let l = 0; l < this.layers.length; l++) {
            const layer = this.layers[l];
            const layerH0 = l === 0 ? h0 : null;
            const H = layer.forward(currentInput, T, layerH0);
            layerActivations.push(H);
            currentInput = H;
        }

        const headOutputs = {};
        for (const head of this.heads) {
            const H = layerActivations[head.layerIndex];
            headOutputs[head.name] = head.forward(H, T);
        }

        this.forwardCache = {
            rawInput,
            T,
            h0,
            condition: options.condition,
            layerActivations,
            headOutputs,
        };

        return headOutputs;
    }

    computeLoss(namedTargets) {
        if (!this.forwardCache) {
            throw new Error("Cannot compute loss before forward pass");
        }
        const { T, headOutputs } = this.forwardCache;
        let totalLoss = 0.0;
        const headResults = {};

        for (const head of this.heads) {
            const targets = namedTargets[head.name];
            if (!targets) continue;

            const out = headOutputs[head.name];
            const probs = out.probs;
            const preds = out.preds;
            const outD = head.outDim;

            let headLoss = 0.0;
            let correct = 0;

            for (let t = 0; t < T; t++) {
                const targetClass = targets[t];
                const p = Math.max(1e-15, probs[t * outD + targetClass]);
                headLoss -= Math.log(p);
                if (preds[t] === targetClass) correct++;
            }

            headLoss /= T;
            const accuracy = correct / T;
            totalLoss += head.lossWeight * headLoss;

            headResults[head.name] = {
                loss: headLoss,
                acc: accuracy,
                weightedLoss: head.lossWeight * headLoss,
            };
        }

        return {
            totalLoss,
            heads: headResults,
        };
    }

    backward(namedTargets) {
        if (!this.forwardCache) {
            throw new Error("Cannot run backward before forward pass");
        }
        const { T, h0, condition, layerActivations, headOutputs } =
            this.forwardCache;

        const dH_layers = this.layers.map((l) =>
            new Float32Array(T * l.outDim)
        );

        for (const head of this.heads) {
            const targets = namedTargets[head.name];
            if (!targets) continue;

            const out = headOutputs[head.name];
            const probs = out.probs;
            const outD = head.outDim;
            const gradLogits = new Float32Array(T * outD);
            const scale = head.lossWeight / T;

            for (let t = 0; t < T; t++) {
                const targetClass = targets[t];
                const offset = t * outD;
                for (let c = 0; c < outD; c++) {
                    const y = c === targetClass ? 1.0 : 0.0;
                    gradLogits[offset + c] = (probs[offset + c] - y) * scale;
                }
            }

            const H = layerActivations[head.layerIndex];
            const dH = dH_layers[head.layerIndex];
            head.backward(gradLogits, H, T, dH);
        }

        let dNextLayerInput = null;
        for (let l = this.layers.length - 1; l >= 0; l--) {
            const layer = this.layers[l];
            const dH = dH_layers[l];
            if (dNextLayerInput) {
                for (let i = 0; i < dH.length; i++) {
                    dH[i] += dNextLayerInput[i];
                }
            }
            const bptt = layer.backward(dH);
            dNextLayerInput = bptt.dX;

            if (l === 0 && this.priming && bptt.dh0_fwd) {
                this.backwardPriming(bptt.dh0_fwd, h0, condition);
            }
        }

        return dNextLayerInput;
    }

    backwardPriming(dh0, h0, condition) {
        if (!this.priming || !dh0) return;
        const H0 = this.layers[0].hiddenDim;
        const pDim = this.priming.inDim;
        const gradW = this.priming.gradW.data;
        const gradB = this.priming.gradB;

        const dPreact = new Float32Array(H0);
        for (let i = 0; i < H0; i++) {
            const dtanh = 1.0 - h0[i] * h0[i];
            dPreact[i] = dh0[i] * dtanh;
            gradB[i] += dPreact[i];
        }

        if (this.priming.type === "categorical") {
            const col = Number(condition);
            if (col >= 0 && col < pDim) {
                for (let i = 0; i < H0; i++) {
                    gradW[i * pDim + col] += dPreact[i];
                }
            }
        } else {
            const condVec = condition;
            for (let i = 0; i < H0; i++) {
                const rowOffset = i * pDim;
                const dP = dPreact[i];
                for (let j = 0; j < pDim; j++) {
                    gradW[rowOffset + j] += dP * condVec[j];
                }
            }
        }
    }

    zeroGrad() {
        for (const p of this.parameters) {
            p.grad.fill(0);
        }
    }

    clipGradients(maxNorm = 5.0) {
        let sumSq = 0.0;
        for (const p of this.parameters) {
            const g = p.grad;
            for (let i = 0; i < g.length; i++) {
                sumSq += g[i] * g[i];
            }
        }
        const totalNorm = Math.sqrt(sumSq);
        if (totalNorm > maxNorm) {
            const scale = maxNorm / (totalNorm + 1e-6);
            for (const p of this.parameters) {
                const g = p.grad;
                for (let i = 0; i < g.length; i++) {
                    g[i] *= scale;
                }
            }
        }
        return totalNorm;
    }

    adamStep(lr = 0.005, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
        this.stepCount++;
        const bc1 = 1.0 - Math.pow(beta1, this.stepCount);
        const bc2 = 1.0 - Math.pow(beta2, this.stepCount);

        for (const p of this.parameters) {
            const w = p.weight;
            const g = p.grad;
            const m = p.mSlice;
            const v = p.vSlice;

            for (let i = 0; i < w.length; i++) {
                const grad = g[i];
                m[i] = beta1 * m[i] + (1.0 - beta1) * grad;
                v[i] = beta2 * v[i] + (1.0 - beta2) * grad * grad;

                const mHat = m[i] / bc1;
                const vHat = v[i] / bc2;

                w[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
            }
        }
    }

    /**
     * Performs a single complete training step:
     * forward, loss, backward, clip gradients, Adam update
     *
     * @param {Float32Array|Array} X - Input features sequence [T x inDim]
     * @param {number} T - Sequence length
     * @param {*} condition - Optional priming condition (scalar or vector)
     * @param {Object} targets - Target classes per head { [headName]: targets }
     * @param {number} [lr=0.005] - Learning rate
     * @returns {Object} Training step results with loss, heads, and gradNorm
     */
    trainStep(X, T, condition, targets, lr = 0.005) {
        this.zeroGrad();
        const options = condition !== undefined && condition !== null
            ? { condition }
            : {};
        const forwardResults = this.forward(X, T, options);
        const lossResults = this.computeLoss(targets);
        this.backward(targets);
        const gradNorm = this.clipGradients(5.0);
        this.adamStep(lr);

        return {
            totalLoss: lossResults.totalLoss,
            heads: lossResults.heads,
            gradNorm,
            forwardResults,
        };
    }

    /**
     * Trains the recurrent network over multiple epochs on a dataset.
     *
     * @param {Array<Object>} dataset - Array of samples:
     *   { features|X|input, length|noteCount|T, condition|initialKey|h0, targets }
     * @param {Object} [options]
     * @param {number} [options.epochs=80] - Number of epochs
     * @param {number} [options.lr=0.01] - Adam learning rate
     * @param {number} [options.logInterval=10] - Log progress every N epochs (0 disables)
     * @param {boolean} [options.shuffle=true] - Shuffle dataset each epoch
     * @param {Function} [options.onEpoch] - Callback ({ epoch, loss, heads }) => void
     * @returns {{ epochs: number, finalLoss: number, heads: Object, duration: number }}
     */
    train(dataset, {
        epochs = 80,
        lr = 0.01,
        logInterval = 10,
        shuffle = true,
        onEpoch = null,
    } = {}) {
        const startTime = performance.now();
        let finalLoss = Infinity;
        const finalHeads = {};

        for (let epoch = 1; epoch <= epochs; epoch++) {
            if (shuffle) {
                for (let i = dataset.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    const temp = dataset[i];
                    dataset[i] = dataset[j];
                    dataset[j] = temp;
                }
            }

            let totalLoss = 0.0;
            const headTotals = {};
            for (const head of this.heads) {
                headTotals[head.name] = { loss: 0.0, acc: 0.0 };
            }

            for (const sample of dataset) {
                const X = sample.features || sample.X || sample.input;
                const T = sample.length ?? sample.noteCount ?? sample.T;
                const condition = sample.condition ?? sample.initialKey ??
                    sample.h0;
                const targets = sample.targets || sample.namedTargets || sample;

                const step = this.trainStep(X, T, condition, targets, lr);

                totalLoss += step.totalLoss;
                for (const head of this.heads) {
                    const hRes = step.heads?.[head.name];
                    if (hRes) {
                        headTotals[head.name].loss += hRes.loss;
                        headTotals[head.name].acc += hRes.acc;
                    }
                }
            }

            const count = dataset.length;
            const avgLoss = totalLoss / count;
            const headAverages = {};
            for (const head of this.heads) {
                headAverages[head.name] = {
                    loss: headTotals[head.name].loss / count,
                    acc: (headTotals[head.name].acc / count) * 100,
                };
            }

            finalLoss = avgLoss;
            Object.assign(finalHeads, headAverages);

            if (onEpoch) {
                onEpoch({ epoch, loss: avgLoss, heads: headAverages });
            }

            if (
                logInterval &&
                (epoch === 1 || epoch % logInterval === 0 || epoch === epochs)
            ) {
                const headLogs = this.heads
                    .filter((h) => headAverages[h.name])
                    .map((h) =>
                        `${h.name} Acc: ${headAverages[h.name].acc.toFixed(1)}%`
                    )
                    .join(" | ");

                console.log(
                    `   Epoch ${String(epoch).padStart(3)} | ` +
                        `Loss: ${avgLoss.toFixed(4)}${
                            headLogs ? " | " + headLogs : ""
                        }`,
                );
            }
        }

        const duration = Number(
            ((performance.now() - startTime) / 1000).toFixed(2),
        );
        return {
            epochs,
            finalLoss,
            heads: finalHeads,
            duration,
        };
    }

    exportWeights() {
        const weights = {};
        for (const p of this.parameters) {
            weights[p.name] = new Float32Array(p.weight);
        }
        return {
            config: {
                inDim: this.inDim,
                layerConfigs: this.layerConfigs,
                primingConfig: this.primingConfig,
                headConfigs: this.headConfigs,
            },
            weights,
        };
    }

    loadWeights(weightsOrArray) {
        if (
            weightsOrArray instanceof Float32Array ||
            weightsOrArray instanceof ArrayBuffer ||
            ArrayBuffer.isView(weightsOrArray)
        ) {
            if (weightsOrArray.byteLength % 4 !== 0) {
                throw new Error(
                    `Invalid weights byte length ${weightsOrArray.byteLength} not a multiple of 4`,
                );
            }
            let flat;
            if (weightsOrArray instanceof Float32Array) {
                flat = weightsOrArray;
            } else {
                const buffer = ArrayBuffer.isView(weightsOrArray)
                    ? weightsOrArray.buffer
                    : weightsOrArray;
                const byteOffset = ArrayBuffer.isView(weightsOrArray)
                    ? weightsOrArray.byteOffset
                    : 0;
                const byteLength = weightsOrArray.byteLength;
                if (byteOffset % 4 === 0) {
                    flat = new Float32Array(buffer, byteOffset, byteLength / 4);
                } else {
                    flat = new Float32Array(
                        buffer.slice(byteOffset, byteOffset + byteLength),
                    );
                }
            }
            const expected = this.parameters.reduce(
                (sum, p) => sum + p.weight.length,
                0,
            );
            if (flat.length !== expected) {
                throw new Error(
                    `Weights length mismatch: got ${flat.length} floats, model expects ${expected}`,
                );
            }
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
}

export { LinearHead, RecurrentNetwork };

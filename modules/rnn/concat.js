/**
 * Shared helpers for resolving and routing extra layer input concatenations
 * (`concat: ["headName", "input", ...]`) across CPU and WebGPU RNNs.
 */

/**
 * Resolves declared output head names, layer indices, and output dimensions
 * before layers are constructed so downstream layers can size `inDim`.
 *
 * @param {Array<Object>} heads - Head config objects or LinearHead instances
 * @param {number} numLayers - Total recurrent layer count
 * @returns {Map<string, { name: string, layerIndex: number, outDim: number }>}
 */
export function resolveHeadMeta(heads, numLayers) {
    const byName = new Map();
    for (let h = 0; h < heads.length; h++) {
        const item = heads[h];
        const name = item.name || `head_${h}`;
        let layerIdx = item.layerIndex !== undefined ? item.layerIndex : -1;
        if (layerIdx < 0) layerIdx = numLayers + layerIdx;
        if (layerIdx < 0 || layerIdx >= numLayers) {
            throw new Error(
                `Head "${name}" layerIndex ${layerIdx} out of range [0, ${numLayers - 1}]`,
            );
        }
        byName.set(name, { name, layerIndex: layerIdx, outDim: item.outDim });
    }
    return byName;
}

/**
 * Resolves and validates a layer's `concat` list into concrete input descriptors.
 *
 * @param {Array<string>|undefined} concatList - Head names or "input"
 * @param {number} layerIndex - Index of the consuming layer
 * @param {number} rawInDim - Network raw input dimension
 * @param {Map} headMetaByName - Map returned by resolveHeadMeta()
 * @returns {Array<{ type: "input"|"head", name: string, dim: number }>}
 */
export function resolveLayerConcat(concatList, layerIndex, rawInDim, headMetaByName) {
    if (!concatList || !concatList.length) return [];

    return concatList.map((name) => {
        if (name === "input") {
            return { type: "input", name: "input", dim: rawInDim };
        }
        const head = headMetaByName.get(name);
        if (!head) {
            throw new Error(
                `Layer ${layerIndex} concat references unknown head "${name}"`,
            );
        }
        if (head.layerIndex >= layerIndex) {
            throw new Error(
                `Layer ${layerIndex} concat head "${name}" must come from an earlier layer (got layerIndex ${head.layerIndex})`,
            );
        }
        return { type: "head", name, dim: head.outDim };
    });
}

/**
 * Packs `[baseInput, ...concatSources]` per timestep on CPU.
 */
export function packConcatCpu(
    baseInput,
    baseDim,
    concatSources,
    rawInput,
    headOutputs,
    T,
    totalInDim,
    outBuf = null,
) {
    const out = outBuf && outBuf.length >= T * totalInDim
        ? outBuf.subarray(0, T * totalInDim)
        : new Float32Array(T * totalInDim);
    for (let t = 0; t < T; t++) {
        const dstRow = t * totalInDim;
        const baseRow = t * baseDim;
        for (let j = 0; j < baseDim; j++) {
            out[dstRow + j] = baseInput[baseRow + j];
        }
        let colOffset = baseDim;
        for (let s = 0; s < concatSources.length; s++) {
            const src = concatSources[s];
            const srcBuf = src.type === "input"
                ? rawInput
                : headOutputs[src.name].probs;
            const srcRow = t * src.dim;
            const dstOffset = dstRow + colOffset;
            for (let j = 0; j < src.dim; j++) {
                out[dstOffset + j] = srcBuf[srcRow + j];
            }
            colOffset += src.dim;
        }
    }
    return out;
}

/**
 * Distributes `dX` from a layer with `concatSources` back to base and source grads.
 */
export function unpackConcatGradCpu(
    dX,
    baseDim,
    concatSources,
    dBaseOut,
    dRawInput,
    dHeadProbs,
    T,
    totalInDim,
) {
    for (let t = 0; t < T; t++) {
        const srcRow = t * totalInDim;
        const baseDst = t * baseDim;
        for (let j = 0; j < baseDim; j++) {
            dBaseOut[baseDst + j] += dX[srcRow + j];
        }
        let colOffset = baseDim;
        for (const src of concatSources) {
            const dstBuf = src.type === "input"
                ? dRawInput
                : (dHeadProbs[src.name] ||
                    (dHeadProbs[src.name] = new Float32Array(T * src.dim)));
            const dstRow = t * src.dim;
            for (let j = 0; j < src.dim; j++) {
                dstBuf[dstRow + j] += dX[srcRow + colOffset + j];
            }
            colOffset += src.dim;
        }
    }
}

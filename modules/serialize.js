/**
 * Model Serialization: Bare Binary Weights & ES Module Exporter
 *
 * Binary format is nothing but the model's parameters as contiguous
 * little-endian Float32 values, in canonical `network.parameters` order.
 * No header, no structure: the network architecture lives in code, and the
 * byte length alone is enough to validate a file against a model.
 */

/**
 * Flattens a network's parameters into a single contiguous Float32Array
 *
 * @param {Object|Float32Array} network - Network with a `parameters` array, or a flat Float32Array
 * @returns {Float32Array} Contiguous float parameter array
 */
export function extractWeightsArray(network) {
    if (network instanceof Float32Array) {
        return network;
    }

    if (!network || !Array.isArray(network.parameters)) {
        throw new Error('serializeWeights: network must expose a `parameters` array');
    }

    const totalFloats = network.parameters.reduce((sum, p) => sum + p.weight.length, 0);
    const flat = new Float32Array(totalFloats);
    let offset = 0;
    for (const p of network.parameters) {
        flat.set(p.weight, offset);
        offset += p.weight.length;
    }
    return flat;
}

/**
 * Serializes network parameter weights into bare Float32 bytes
 *
 * @param {Object|Float32Array} network - Model or flat weights
 * @returns {Uint8Array} Bare weight bytes
 */
export function serializeWeights(network) {
    const flat = extractWeightsArray(network);
    return new Uint8Array(flat.buffer.slice(flat.byteOffset, flat.byteOffset + flat.byteLength));
}

/**
 * Encodes binary weights into an exportable, self-contained JavaScript ES Module
 * using Uint8Array.prototype.toBase64 and Uint8Array.fromBase64
 *
 * @param {Object} network - SpellerNetwork or RecurrentNetwork
 * @param {string} importPath - Relative import path to deserialize.js
 * @returns {string} JavaScript module code
 */
export function exportAsModule(network) {
    const binary = serializeWeights(network);
    const base64 = binary.toBase64();

    return `// Auto-generated Pre-Trained Weights
const bytes = Uint8Array.fromBase64("${ base64 }");
export default new Float32Array(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength / 4
);
`;
}

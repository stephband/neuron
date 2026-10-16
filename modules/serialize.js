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
 * Serializes network parameter weights into binary bytes (Float32, Float16, or Int8 with scale header)
 *
 * @param {Object|Float32Array} network - Model or flat weights
 * @param {number} [bits=32] - Bit width per weight: 32 (Float32), 16 (Float16), or 8 (Int8 + 4-byte Float32 scale header)
 * @returns {Uint8Array} Serialized weight bytes
 */
export function serializeWeights(network, bits = 32) {
    const flat = extractWeightsArray(network);

    if (bits === 8) {
        let maxAbs = 0;
        for (let i = 0; i < flat.length; i++) {
            const a = Math.abs(flat[i]);
            if (a > maxAbs) maxAbs = a;
        }
        const scale = maxAbs / 127 || 1;
        const binary = new Uint8Array(4 + flat.length);
        new DataView(binary.buffer).setFloat32(0, scale, true);
        const int8 = new Int8Array(binary.buffer, 4, flat.length);
        for (let i = 0; i < flat.length; i++) {
            int8[i] = Math.max(-127, Math.min(127, Math.round(flat[i] / scale)));
        }
        return binary;
    }

    if (bits === 16) {
        const f16 = new Float16Array(flat);
        return new Uint8Array(f16.buffer, f16.byteOffset, f16.byteLength);
    }

    return new Uint8Array(
        flat.buffer.slice(flat.byteOffset, flat.byteOffset + flat.byteLength),
    );
}

/**
 * Encodes binary weights into an exportable, self-contained JavaScript ES Module
 * using Uint8Array.prototype.toBase64 and Uint8Array.fromBase64
 *
 * @param {Object|Float32Array} network - RecurrentNetwork or flat weights
 * @param {number} [bits=32] - Bit width per weight: 32, 16, or 8
 * @returns {string} JavaScript module code
 */
export function exportAsModule(network, bits = 32) {
    if (bits === 8) {
        const flat = extractWeightsArray(network);
        let maxAbs = 0;
        for (let i = 0; i < flat.length; i++) {
            const a = Math.abs(flat[i]);
            if (a > maxAbs) maxAbs = a;
        }
        const scale = maxAbs / 127 || 1;
        const i8 = new Int8Array(flat.length);
        for (let i = 0; i < flat.length; i++) {
            i8[i] = Math.max(-127, Math.min(127, Math.round(flat[i] / scale)));
        }
        const base64 = new Uint8Array(i8.buffer, i8.byteOffset, i8.byteLength).toBase64();

        return `// Auto-generated Pre-Trained Weights (Int8 quantized)
const scale = ${scale};
const bytes = Uint8Array.fromBase64("${base64}");
const int8  = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
export default Float32Array.from(int8, (x) => x * scale);
`;
    }

    const binary = serializeWeights(network, bits);
    const base64 = binary.toBase64();

    if (bits === 16) {
        return `// Auto-generated Pre-Trained Weights (Float16)
const bytes = Uint8Array.fromBase64("${base64}");
export default new Float32Array(new Float16Array(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength / 2
));
`;
    }

    return `// Auto-generated Pre-Trained Weights
const bytes = Uint8Array.fromBase64("${base64}");
export default new Float32Array(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength / 4
);
`;
}

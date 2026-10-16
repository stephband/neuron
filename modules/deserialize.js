/**
 * Deserializes and normalizes model weights into a flat Float32Array.
 * Supports:
 * - Float32Array (returned directly)
 * - Float16Array (converted to Float32Array)
 * - ArrayBuffer / TypedArray views of:
 *   - Float32 weights (expectedParams * 4 bytes)
 *   - Float16 weights (expectedParams * 2 bytes)
 *   - Int8 quantized weights with a 4-byte little-endian Float32 scale header (expectedParams + 4 bytes)
 *
 * @param {Float32Array|Float16Array|ArrayBuffer|ArrayBufferView} data - Weight data
 * @param {number} [expectedParams] - Expected number of float parameters in the model
 * @returns {Float32Array} Flat parameter weights
 */
export function normalizeWeights(data, expectedParams) {
    if (data instanceof Float32Array) {
        if (expectedParams && data.length !== expectedParams) {
            throw new Error(
                `Weights length mismatch: got ${data.length} floats, model expects ${expectedParams}`,
            );
        }
        return data;
    }

    if (typeof Float16Array !== "undefined" && data instanceof Float16Array) {
        if (expectedParams && data.length !== expectedParams) {
            throw new Error(
                `Weights length mismatch: got ${data.length} floats, model expects ${expectedParams}`,
            );
        }
        return new Float32Array(data);
    }

    if (!(data instanceof ArrayBuffer) && !ArrayBuffer.isView(data)) {
        throw new Error("Invalid weights: expected TypedArray or ArrayBuffer");
    }

    const buffer = ArrayBuffer.isView(data) ? data.buffer : data;
    const byteOffset = ArrayBuffer.isView(data) ? data.byteOffset : 0;
    const byteLength = data.byteLength;

    if (expectedParams) {
        // 8-bit quantized format: 4-byte Float32 scale header + Int8Array[expectedParams]
        if (byteLength === expectedParams + 4) {
            const scale = new DataView(buffer, byteOffset, 4).getFloat32(
                0,
                true,
            );
            const int8 = new Int8Array(buffer, byteOffset + 4, expectedParams);
            return Float32Array.from(int8, (x) => x * scale);
        }

        // 16-bit float format: Float16Array[expectedParams]
        if (
            typeof Float16Array !== "undefined" &&
            byteLength === expectedParams * 2
        ) {
            const f16 = byteOffset % 2 === 0
                ? new Float16Array(buffer, byteOffset, expectedParams)
                : new Float16Array(
                    buffer.slice(byteOffset, byteOffset + byteLength),
                );
            return new Float32Array(f16);
        }

        // 32-bit float format: Float32Array[expectedParams]
        if (byteLength === expectedParams * 4) {
            return byteOffset % 4 === 0
                ? new Float32Array(buffer, byteOffset, expectedParams)
                : new Float32Array(
                    buffer.slice(byteOffset, byteOffset + byteLength),
                );
        }

        throw new Error(
            `Weights byte length mismatch: got ${byteLength} bytes, model expects ${
                expectedParams * 4
            } bytes (32-bit), ${expectedParams * 2} bytes (16-bit), or ${
                expectedParams + 4
            } bytes (8-bit)`,
        );
    }

    if (byteLength % 4 !== 0) {
        throw new Error(
            `Invalid weights: byte length ${byteLength} is not a multiple of 4`,
        );
    }

    // Ensure 4-byte alignment
    return byteOffset % 4 === 0
        ? new Float32Array(buffer, byteOffset, byteLength / 4)
        : new Float32Array(buffer.slice(byteOffset, byteOffset + byteLength));
}

export default normalizeWeights;

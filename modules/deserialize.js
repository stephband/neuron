/**
 * Deserializes bare Float32 model weights from binary data (ArrayBuffer or Uint8Array).
 *
 * @param {ArrayBuffer|Uint8Array|Float32Array} data - Binary weight data
 * @returns {Float32Array} Flat parameter weights
 */
export default function deserializeWeights(data) {
    if (data instanceof Float32Array) {
        return data;
    }

    const buffer     = ArrayBuffer.isView(data) ? data.buffer : data;
    const byteOffset = ArrayBuffer.isView(data) ? data.byteOffset : 0;
    const byteLength = data.byteLength;

    if (byteLength % 4 !== 0) {
        throw new Error(`Invalid weights: byte length ${byteLength} is not a multiple of 4`);
    }

    // Ensure 4-byte alignment
    return byteOffset % 4 === 0 ?
        new Float32Array(buffer, byteOffset, byteLength / 4) :
        new Float32Array(buffer.slice(byteOffset, byteOffset + byteLength)) ;
}

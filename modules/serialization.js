/**
 * Model Serialization: Compact Binary & ES Module Weight Exporter/Loader
 * 
 * Format:
 * [0..3]:   Magic Bytes "NEUR" (0x4E, 0x45, 0x55, 0x52)
 * [4..7]:   Format Version (Uint32 = 1)
 * [8..11]:  Header Length in Bytes (Uint32 = H_len)
 * [12..12+H_len-1]: UTF-8 JSON Header (architecture config & tensor offsets)
 * [aligned]: Packed Float32 binary data
 */

const MAGIC = 0x5255454e; // "NEUR" in little-endian Uint32
const VERSION = 1;

/**
 * Serializes network architecture and parameters into a contiguous Uint8Array
 * 
 * @param {Object} network - RecurrentNetwork or SpellerNetwork
 * @returns {Uint8Array} Packed binary buffer
 */
export function serializeWeights(network) {
  const exported = network.exportWeights ? network.exportWeights() : network;
  const { weights, ...config } = exported;

  // 1. Calculate tensor byte offsets and total payload size
  const tensorManifest = {};
  let currentOffset = 0;

  for (const [name, array] of Object.entries(weights)) {
    const floatArray = array instanceof Float32Array ? array : new Float32Array(array);
    const count = floatArray.length;
    tensorManifest[name] = {
      offset: currentOffset,
      length: count,
      byteLength: count * 4,
    };
    currentOffset += count * 4;
  }

  // 2. Encode Header JSON
  const headerObj = {
    config,
    tensors: tensorManifest,
    totalBytes: currentOffset,
  };
  const headerJson = JSON.stringify(headerObj);
  const headerBytes = new TextEncoder().encode(headerJson);
  const headerLen = headerBytes.length;

  // 4-byte aligned padding for header
  const paddedHeaderLen = Math.ceil(headerLen / 4) * 4;
  const totalBufferSize = 12 + paddedHeaderLen + currentOffset;

  const buffer = new ArrayBuffer(totalBufferSize);
  const dataView = new DataView(buffer);
  const u8View = new Uint8Array(buffer);

  // 3. Write binary header: Magic, Version, HeaderLen
  dataView.setUint32(0, MAGIC, true);
  dataView.setUint32(4, VERSION, true);
  dataView.setUint32(8, headerLen, true);

  // Write JSON string
  u8View.set(headerBytes, 12);

  // 4. Write Float32 tensor payloads
  const payloadByteStart = 12 + paddedHeaderLen;
  for (const [name, info] of Object.entries(tensorManifest)) {
    const tensorData = weights[name];
    const floatArray = tensorData instanceof Float32Array ? tensorData : new Float32Array(tensorData);
    const targetU8 = new Uint8Array(buffer, payloadByteStart + info.offset, info.byteLength);
    const sourceU8 = new Uint8Array(floatArray.buffer, floatArray.byteOffset, floatArray.byteLength);
    targetU8.set(sourceU8);
  }

  return u8View;
}

/**
 * Deserializes network configuration and parameters from a binary buffer
 * 
 * @param {ArrayBuffer|Uint8Array} binaryData
 * @returns {Object} { config, weights }
 */
export function deserializeWeights(binaryData) {
  const buffer = binaryData instanceof Uint8Array ? binaryData.buffer : binaryData;
  const byteOffset = binaryData instanceof Uint8Array ? binaryData.byteOffset : 0;
  const byteLength = binaryData.byteLength;

  const dataView = new DataView(buffer, byteOffset, byteLength);
  const magic = dataView.getUint32(0, true);
  if (magic !== MAGIC) {
    throw new Error(`Invalid model binary format: magic 0x${magic.toString(16)} != 0x${MAGIC.toString(16)}`);
  }

  const version = dataView.getUint32(4, true);
  if (version !== VERSION) {
    throw new Error(`Unsupported model format version: ${version}`);
  }

  const headerLen = dataView.getUint32(8, true);
  const paddedHeaderLen = Math.ceil(headerLen / 4) * 4;

  const u8View = new Uint8Array(buffer, byteOffset, byteLength);
  const headerBytes = u8View.subarray(12, 12 + headerLen);
  const headerJson = new TextDecoder().decode(headerBytes);
  const headerObj = JSON.parse(headerJson);

  const payloadByteStart = 12 + paddedHeaderLen;
  const weights = {};

  for (const [name, info] of Object.entries(headerObj.tensors)) {
    const tensorBytes = buffer.slice(
      byteOffset + payloadByteStart + info.offset,
      byteOffset + payloadByteStart + info.offset + info.byteLength
    );
    weights[name] = new Float32Array(tensorBytes);
  }

  return {
    ...headerObj.config,
    weights,
  };
}

/**
 * Encodes binary weights into an exportable, self-contained JavaScript ES Module
 * using Uint8Array.prototype.toBase64 and Uint8Array.fromBase64
 * 
 * @param {Object} network - SpellerNetwork or RecurrentNetwork
 * @param {string} importPath - Relative import path to serialization.js
 * @returns {string} JavaScript module code
 */
export function exportAsModule(network, importPath = '../modules/serialization.js') {
  const binary = serializeWeights(network);
  const base64 = binary.toBase64();

  return `// Auto-generated Pre-Trained Weights
import { deserializeWeights } from '${importPath}';

const BASE64_DATA = "${base64}";

let cachedWeights = null;

export function getTrainedModelWeights() {
  if (!cachedWeights) {
    const u8 = Uint8Array.fromBase64(BASE64_DATA);
    cachedWeights = deserializeWeights(u8);
  }
  return cachedWeights;
}
`;
}

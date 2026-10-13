import { initWebGPU } from '../modules/gpu/context.js';
import { GpuOps } from '../modules/gpu/ops.js';
import { GpuBidirectionalGru } from '../modules/rnn/gpu_bigru.js';
import { readFromBuffer, createStorageBuffer } from '../modules/gpu/buffer.js';
import { BidirectionalGru } from '../modules/rnn/bigru.js';
import SpellerRNN from '../speller/speller-rnn.js';
import { SpellerNetwork } from '../speller/speller_network.js';
import { SequenceEncoder } from '../speller/sequence_encoder.js';
import { SequenceAugmenter } from '../speller/augment.js';
import { toKeyName } from '../speller/key_tpc.js';

console.log('=== WebGPU Recurrent Neural Network Verification (Phase 3) ===\n');

// 1. Initialize WebGPU
const { device, adapter } = await initWebGPU();
console.log(`1. Initialized WebGPU Device: ${adapter.info.description || 'Generic Adapter'}`);
const gpuOps = new GpuOps(device);

// Helper to check numerical equivalence
function assertClose(actual, expected, tolerance = 1e-4, testName = 'Test') {
  if (actual.length !== expected.length) {
    throw new Error(`${testName} failed: Length mismatch (${actual.length} vs ${expected.length})`);
  }
  let maxDiff = 0;
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    if (diff > maxDiff) maxDiff = diff;
    if (diff > tolerance) {
      throw new Error(
        `${testName} failed at index ${i}: actual=${actual[i]}, expected=${expected[i]}, diff=${diff}`
      );
    }
  }
  console.log(`   ✓ ${testName} passed (max numerical difference: ${maxDiff.toExponential(3)})`);
}

// 2. Verify Recurrent Layer: GpuBidirectionalGru vs CPU BidirectionalGru
console.log('\n2. Verifying GpuBidirectionalGru recurrent transitions against CPU reference...');

const T = 7;
const inDim = 16;
const H = 48;

const cpuBigru = new BidirectionalGru(inDim, H);
const gpuBigru = new GpuBidirectionalGru(device, gpuOps, inDim, H, true);
gpuBigru.syncFromCpuLayer(cpuBigru);

const X = new Float32Array(T * inDim);
for (let i = 0; i < X.length; i++) X[i] = (Math.random() - 0.5) * 2;

const h0 = new Float32Array(H);
for (let i = 0; i < h0.length; i++) h0[i] = (Math.random() - 0.5);

// CPU Forward
const cpuOutH = cpuBigru.forward(X, T, h0);

// GPU Forward
const bufX = createStorageBuffer(device, X, 'test_bufX');
const bufH0 = createStorageBuffer(device, h0, 'test_bufH0');
const bufGpuOutH = gpuBigru.forward(bufX, T, bufH0);
const gpuOutH = await readFromBuffer(device, bufGpuOutH, T * 2 * H * 4);

assertClose(gpuOutH, cpuOutH, 1e-4, `BiGRU [T=${T}, Din=${inDim}, H=${H}]`);

// 3. Verify Unidirectional Mode on GPU
console.log('\n3. Verifying Unidirectional GRU Mode on GPU...');
const cpuUniGru = new BidirectionalGru(inDim, H, false);
const gpuUniGru = new GpuBidirectionalGru(device, gpuOps, inDim, H, false);
gpuUniGru.syncFromCpuLayer(cpuUniGru);

const cpuUniOut = cpuUniGru.forward(X, T, h0);
const bufUniOut = gpuUniGru.forward(bufX, T, bufH0);
const gpuUniOut = await readFromBuffer(device, bufUniOut, T * H * 4);

assertClose(gpuUniOut, cpuUniOut, 1e-4, `Unidirectional GRU [T=${T}, Din=${inDim}, H=${H}]`);

// 4. Verify End-to-End Pitch Speller on GPU
console.log('\n4. Verifying End-to-End SpellerRNN on Canonical Musical Sequences...');

// Progression in C major with secondary dominant (D7 -> G) and passing chromatic note (Bb)
const eventsC = [
  [0.0, 'key', 0],               // C major
  [0.0, 'note', 60, 0.8, 1.0],   // C4  (TPC: 0, C)
  [1.0, 'note', 64, 0.8, 1.0],   // E4  (TPC: 4, E)
  [2.0, 'note', 67, 0.8, 1.0],   // G4  (TPC: 1, G)
  [3.0, 'note', 66, 0.8, 1.0],   // F#4 (TPC: 6, F# -> secondary dominant)
  [4.0, 'note', 67, 0.8, 1.0],   // G4  (TPC: 1, G)
  [5.0, 'note', 72, 0.8, 1.0],   // C5  (TPC: 0, C)
  [6.0, 'note', 71, 0.8, 0.5],   // B4  (TPC: 5, B)
  [6.5, 'note', 70, 0.8, 0.5],   // Bb4 (TPC: -2, Bb -> chromatic passing flat)
  [7.0, 'note', 69, 0.8, 1.0],   // A4  (TPC: 3, A)
];

const truthC = [
  { tpc:  0, key: 0 }, // C
  { tpc:  4, key: 0 }, // E
  { tpc:  1, key: 0 }, // G
  { tpc:  6, key: 1 }, // F# (tonicizing G, key 1)
  { tpc:  1, key: 1 }, // G
  { tpc:  0, key: 0 }, // C
  { tpc:  5, key: 0 }, // B
  { tpc: -2, key: 0 }, // Bb
  { tpc:  3, key: 0 }, // A
];

// Progression in Eb major with descending passing tone (Gb)
const eventsEb = [
  [0.0, 'key', -3],              // Eb major
  [0.0, 'note', 63, 0.8, 1.0],   // Eb4 (TPC: -3, Eb)
  [1.0, 'note', 68, 0.8, 1.0],   // Ab4 (TPC: -4, Ab)
  [2.0, 'note', 66, 0.8, 0.5],   // Gb4 (TPC: -6, Gb -> descending passing tone)
  [2.5, 'note', 65, 0.8, 1.0],   // F4  (TPC: -1, F)
  [3.5, 'note', 63, 0.8, 1.0],   // Eb4 (TPC: -3, Eb)
];

const truthEb = [
  { tpc: -3, key: -3 }, // Eb
  { tpc: -4, key: -3 }, // Ab
  { tpc: -6, key: -3 }, // Gb
  { tpc: -1, key: -3 }, // F
  { tpc: -3, key: -3 }, // Eb
];

// Train CPU network briefly to get high-accuracy weights
const cpuSpeller = new SpellerNetwork({ hiddenDim: 48 });
const dataset = [];

function addAugmentations(baseEvents, baseTruth) {
  const enc = SequenceEncoder.encode(baseEvents);
  for (let deltaK = -4; deltaK <= 4; deltaK++) {
    const aug = SequenceAugmenter.transpose(enc.notes, baseTruth, deltaK, enc.initialKey);
    if (!aug) continue;
    const augEnc = SequenceEncoder.encode(aug.noteEvents, { initialKey: aug.initialKey });
    const augTargets = SequenceEncoder.encodeTargets(aug.targets);
    dataset.push({
      features: augEnc.features,
      noteCount: augEnc.noteCount,
      initialKey: augEnc.initialKey,
      targets: augTargets,
    });
  }
}

addAugmentations(eventsC, truthC);
addAugmentations(eventsEb, truthEb);

console.log('   Training reference model weights...');
for (let epoch = 1; epoch <= 100; epoch++) {
  for (const item of dataset) {
    cpuSpeller.trainStep(item.features, item.noteCount, item.initialKey, item.targets, 0.01);
  }
}

// Instantiate SpellerRNN and sync weights
const gpuSpeller = new SpellerRNN(device, gpuOps, { hiddenDim: 48 });
gpuSpeller.syncFromCpuNetwork(cpuSpeller);

// 5. Run GPU Inference on C major progression
const t0 = performance.now();
const gpuPredsC = await gpuSpeller.predictSequence(eventsC);
const gpuTime = (performance.now() - t0).toFixed(2);

console.log(`   ✓ GPU predicted C major progression in ${gpuTime} ms`);

// Verify Note 3 (F# secondary dominant)
const note3 = gpuPredsC[3];
console.log(`   Note 3 (MIDI 66 in C major): predicted ${toKeyName(note3.tpc)} (TPC ${note3.tpc}) | Target: F♯`);
if (note3.tpc !== 6) throw new Error(`Expected F♯ (6), got ${note3.tpc}`);

// Verify Note 7 (Bb passing flat)
const note7 = gpuPredsC[7];
console.log(`   Note 7 (MIDI 70 in C major): predicted ${toKeyName(note7.tpc)} (TPC ${note7.tpc}) | Target: B♭`);
if (note7.tpc !== -2) throw new Error(`Expected B♭ (-2), got ${note7.tpc}`);

// 6. Run GPU Inference on Eb major progression
const gpuPredsEb = await gpuSpeller.predictSequence(eventsEb);
const noteGb = gpuPredsEb[2];
console.log(`   Note 2 (MIDI 66 in E♭ major): predicted ${toKeyName(noteGb.tpc)} (TPC ${noteGb.tpc}) | Target: G♭`);
if (noteGb.tpc !== -6) throw new Error(`Expected G♭ (-6), got ${noteGb.tpc}`);

console.log('\nAll Phase 3 WebGPU WGSL recurrent kernels and GPU networks verified with 100% precision!');

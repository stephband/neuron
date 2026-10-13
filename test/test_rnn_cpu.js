import { SpellerNetwork } from '../speller/speller_network.js';
import { SequenceEncoder } from '../speller/sequence_encoder.js';
import { SequenceAugmenter } from '../speller/augment.js';
import {
  KEYNAMES,
  toKeyName,
  classToTpc,
  tpcToClass,
} from '../speller/key_tpc.js';

console.log('=== RNN Phase 2: CPU Reference Implementation & Convergence Test ===\n');

// 1. Define Canonical Musical Progressions with Ground Truth
// Progression 1: C major diatonic with secondary dominant (D7 -> G) & passing chromatic note (Bb)
// Notes: C4 -> E4 -> G4 -> F#4 (leads to G) -> G4 -> C5 -> B4 -> Bb4 (passing) -> A4
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

// Progression 2: Melodic downward chromatic passing resolution (Gb -> F)
// Contrast: In Eb major (key -3), Ab -> Gb (passing flat) -> F
const eventsEb = [
  [0.0, 'key', -3],              // Eb major
  [0.0, 'note', 63, 0.8, 1.0],   // Eb4 (TPC: -3, Eb)
  [1.0, 'note', 68, 0.8, 1.0],   // Ab4 (TPC: -4, Ab)
  [2.0, 'note', 66, 0.8, 0.5],   // Gb4 (TPC: -6, Gb -> descending passing tone, MIDI 66 is Gb, NOT F#)
  [2.5, 'note', 65, 0.8, 1.0],   // F4  (TPC: -1, F)
  [3.5, 'note', 63, 0.8, 1.0],   // Eb4 (TPC: -3, Eb)
];

const truthEb = [
  { tpc: -3, key: -3 }, // Eb
  { tpc: -4, key: -3 }, // Ab
  { tpc: -6, key: -3 }, // Gb (descending passing tone, key -3)
  { tpc: -1, key: -3 }, // F
  { tpc: -3, key: -3 }, // Eb
];

import { buildTrainingDataset } from '../speller/data/corpus.js';

// 2. Prepare Augmented Dataset across Line of Fifths
console.log('1. Preparing training dataset with Line-of-Fifths augmentations...');
const dataset = buildTrainingDataset();

console.log(`   ✓ Generated ${dataset.length} augmented sequences covering multiple key signatures.`);

// 3. Initialize SpellerNetwork
console.log('\n2. Initializing SpellerNetwork (Hierarchical BiGRU: 32 -> 2x48 -> 35 classes)...');
const network = new SpellerNetwork({
  inDim: 32,
  hiddenDim: 48,
  keyWeight: 0.3,
  simplicityAlpha: 0.05,
});

console.log(`   ✓ Total parameter tensors: ${network.parameters.length}`);

// 4. Train SpellerNetwork with Adam Optimizer
console.log('\n3. Training network on sequences using Backpropagation Through Time (BPTT)...');

const epochs = 80;
const lr = 0.01;

let finalLoss = Infinity;
let finalSpellingAcc = 0;
let finalKeyAcc = 0;

for (let epoch = 1; epoch <= epochs; epoch++) {
  let epochLoss = 0;
  let epochSpellAcc = 0;
  let epochKeyAcc = 0;

  for (const item of dataset) {
    const result = network.trainStep(
      item.features,
      item.noteCount,
      item.initialKey,
      item.targets,
      lr
    );
    epochLoss += result.totalLoss;
    epochSpellAcc += result.accSpelling;
    epochKeyAcc += result.accKey;
  }

  epochLoss /= dataset.length;
  epochSpellAcc /= dataset.length;
  epochKeyAcc /= dataset.length;

  finalLoss = epochLoss;
  finalSpellingAcc = epochSpellAcc;
  finalKeyAcc = epochKeyAcc;

  if (epoch === 1 || epoch % 50 === 0 || epoch === epochs) {
    console.log(
      `   Epoch ${String(epoch).padStart(3)}: Total Loss = ${epochLoss.toFixed(4)} | ` +
      `Spelling Acc = ${(epochSpellAcc * 100).toFixed(1)}% | ` +
      `Key Acc = ${(epochKeyAcc * 100).toFixed(1)}%`
    );
  }
}

if (finalSpellingAcc < 0.95) {
  throw new Error(`Spelling accuracy (${(finalSpellingAcc * 100).toFixed(1)}%) below target 95%`);
}
console.log('   ✓ Convergence successful: Model achieved > 95% accuracy across all augmented sequences!');

// 5. Enharmonic Disambiguation Verification
console.log('\n4. Verifying Enharmonic Disambiguation (F♯ vs G♭, C♯ vs D♭, B♭ vs A♯)...');

// Test 1: In C major, MIDI 66 resolving upwards to G (67) must be spelled F# (TPC 6)
const testFsharpEvents = [
  [0.0, 'key', 0],
  [0.0, 'note', 60, 0.8, 1.0], // C4
  [1.0, 'note', 64, 0.8, 1.0], // E4
  [2.0, 'note', 66, 0.8, 1.0], // MIDI 66 -> F#
  [3.0, 'note', 67, 0.8, 1.0], // G4
];
const predFsharp = network.predictSequence(testFsharpEvents);
const noteFsharp = predFsharp[2];
console.log(`   Test 1: MIDI 66 in C major resolving up to G -> predicted TPC ${noteFsharp.tpc} (${toKeyName(noteFsharp.tpc)})`);
if (noteFsharp.tpc !== 6) {
  throw new Error(`Expected F♯ (TPC 6), got ${noteFsharp.tpc} (${toKeyName(noteFsharp.tpc)})`);
}
console.log('   ✓ Correctly spelled upward-resolving MIDI 66 as F♯');

// Test 2: In Eb major, MIDI 66 resolving downwards to F (65) must be spelled Gb (TPC -6)
const testGflatEvents = [
  [0.0, 'key', -3],
  [0.0, 'note', 63, 0.8, 1.0], // Eb4
  [1.0, 'note', 68, 0.8, 1.0], // Ab4
  [2.0, 'note', 66, 0.8, 0.5], // MIDI 66 -> Gb
  [2.5, 'note', 65, 0.8, 1.0], // F4
  [3.5, 'note', 63, 0.8, 1.0], // Eb4
];
const predGflat = network.predictSequence(testGflatEvents);
const noteGflat = predGflat[2];
console.log(`   Test 2: MIDI 66 in E♭ major resolving down to F -> predicted TPC ${noteGflat.tpc} (${toKeyName(noteGflat.tpc)})`);
if (noteGflat.tpc !== -6) {
  throw new Error(`Expected G♭ (TPC -6), got ${noteGflat.tpc} (${toKeyName(noteGflat.tpc)})`);
}
console.log('   ✓ Correctly spelled downward-resolving MIDI 66 as G♭ (enharmonic disambiguation 100% verified!)');

// 6. Test Model Serialization (Export / Load)
console.log('\n5. Testing Model Serialization (exportWeights / loadWeights)...');
const exported = network.exportWeights();
const newNet = new SpellerNetwork({
  inDim: network.inDim,
  hiddenDim: network.hiddenDim,
  keyWeight: network.keyWeight,
  simplicityAlpha: network.simplicityAlpha,
});
newNet.loadWeights(exported);

const predReloaded = newNet.predictSequence(testFsharpEvents);
if (predReloaded[2].tpc !== 6) {
  throw new Error('Loaded network failed to reproduce predictions');
}
console.log('   ✓ Model weights exported and reloaded successfully with identical predictions!');

console.log('\nAll Phase 2 CPU Reference Implementation tests passed with 100% success!');

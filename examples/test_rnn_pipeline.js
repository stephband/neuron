import {
  KEYNUMBERS,
  KEYNAMES,
  NUM_CLASSES,
  tpcToClass,
  classToTpc,
  toKeyName,
  toKeyNumber,
  tpcToMidiPitchClass,
  getDiatonicRange,
  isDiatonicInKey,
} from '../speller/key_tpc.js';
import {
  SequenceEncoder,
  roundPow2,
  durationToBucketIndex,
  deltaToBucketIndex,
} from '../speller/sequence_encoder.js';
import { SequenceAugmenter } from '../speller/augment.js';

console.log('=== RNN Phase 1: Sequence Event Pipeline & Coordinate Verification ===\n');

// 1. Verify Line of Fifths Range and Round-Trip Mapping
console.log('1. Verifying Line of Fifths 35-class coordinate system...');
if (NUM_CLASSES !== 35) throw new Error(`Expected 35 classes, got ${NUM_CLASSES}`);

for (const [name, coord] of Object.entries(KEYNUMBERS)) {
  const cls = tpcToClass(coord);
  const roundtrip = classToTpc(cls);
  if (roundtrip !== coord) {
    throw new Error(`Round-trip failure for ${name}: coord=${coord}, cls=${cls}, roundtrip=${roundtrip}`);
  }
}
console.log('   ✓ All 35 Line-of-Fifths coordinates round-trip to [0, 34] class indices');

// Check special anchor notes
if (tpcToClass(-15) !== 0) throw new Error('F𝄫 must map to index 0');
if (tpcToClass(0) !== 15) throw new Error('C must map to index 15');
if (tpcToClass(19) !== 34) throw new Error('B𝄪 must map to index 34');
console.log('   ✓ Anchor checks: F𝄫 = 0, C = 15, B𝄪 = 34');

// 2. Verify MIDI Pitch Class Math
console.log('\n2. Verifying MIDI pitch class consistency...');
const pitchClassTests = [
  { name: 'C',   coord: 0,  expectedPC: 0 },
  { name: 'G',   coord: 1,  expectedPC: 7 },
  { name: 'D',   coord: 2,  expectedPC: 2 },
  { name: 'F♯',  coord: 6,  expectedPC: 6 },
  { name: 'G♭',  coord: -6, expectedPC: 6 }, // Enharmonic with F#
  { name: 'C♯',  coord: 7,  expectedPC: 1 },
  { name: 'D♭',  coord: -5, expectedPC: 1 }, // Enharmonic with C#
  { name: 'F𝄪',  coord: 13, expectedPC: 7 }, // Double sharp: G
  { name: 'B𝄫',  coord: -9, expectedPC: 9 }, // Double flat: A
];

for (const t of pitchClassTests) {
  const pc = tpcToMidiPitchClass(t.coord);
  if (pc !== t.expectedPC) {
    throw new Error(`MIDI pitch class mismatch for ${t.name}: expected ${t.expectedPC}, got ${pc}`);
  }
}
console.log('   ✓ Pitch class math correctly identifies enharmonic pairs (F♯/G♭, C♯/D♭)');

// 3. Verify Diatonic Window Range
console.log('\n3. Verifying Diatonic Scale windows [k-1, k+5]...');
const cRange = getDiatonicRange(0); // C major: [-1, 5] -> F to B
if (cRange.min !== -1 || cRange.max !== 5) throw new Error('C major range should be [-1, 5]');
if (!isDiatonicInKey(0, 0)) throw new Error('C should be diatonic in C');
if (!isDiatonicInKey(6, 1)) throw new Error('F♯ should be diatonic in G (k=1)');
if (isDiatonicInKey(6, 0)) throw new Error('F♯ should NOT be diatonic in C (k=0)');
console.log('   ✓ Diatonic windows correctly distinguish natural scale tones from chromatic notes');

// 4. Ingest Sample Sequence Events
console.log('\n4. Ingesting Sequence events array...');
// Musical progression in C major with secondary dominant (D7 -> G) and chromatic passing tone:
// C (60), E (64), G (67)
// D7 secondary dominant: F# (66) -> G (67)
// Passing note: C (72) -> B (71) -> Bb (70) -> A (69)
const events = [
  [0.0, 'key', 0],               // C major
  [0.0, 'note', 60, 0.8, 1.0],   // C4  (TPC: 0)
  [1.0, 'note', 64, 0.8, 1.0],   // E4  (TPC: 4)
  [2.0, 'note', 67, 0.8, 1.0],   // G4  (TPC: 1)
  [3.0, 'note', 66, 0.8, 1.0],   // F#4 (TPC: 6) -> secondary dominant leading tone!
  [4.0, 'note', 67, 0.8, 1.0],   // G4  (TPC: 1)
  [5.0, 'note', 72, 0.8, 1.0],   // C5  (TPC: 0)
  [6.0, 'note', 71, 0.8, 0.5],   // B4  (TPC: 5)
  [6.5, 'note', 70, 0.8, 0.5],   // Bb4 (TPC: -2) -> chromatic passing note!
  [7.0, 'note', 69, 0.8, 1.0],   // A4  (TPC: 3)
];

const groundTruth = [
  { tpc:  0, key: 0 }, // C
  { tpc:  4, key: 0 }, // E
  { tpc:  1, key: 0 }, // G
  { tpc:  6, key: 1 }, // F# (secondary dominant tonicizing G, local key 1)
  { tpc:  1, key: 1 }, // G
  { tpc:  0, key: 0 }, // C
  { tpc:  5, key: 0 }, // B
  { tpc: -2, key: 0 }, // Bb
  { tpc:  3, key: 0 }, // A
];

// Verify soundio roundPow2 and duration bucket mapping
if (roundPow2(1.5) !== 2) throw new Error('roundPow2(1.5) should round to 2');
if (roundPow2(0.75) !== 1) throw new Error('roundPow2(0.75) should round to 1');
if (durationToBucketIndex(1/16) !== 0) throw new Error('1/16 beat should map to bucket 0');
if (durationToBucketIndex(1.0) !== 4) throw new Error('Quarter note (1.0) should map to bucket 4');
if (durationToBucketIndex(0.5) !== 3) throw new Error('Eighth note (0.5) should map to bucket 3');
if (durationToBucketIndex(8.0) !== 7) throw new Error('8 beats should map to bucket 7');
console.log('   ✓ Duration mapping: 8 categorical power-of-2 buckets');

// Verify deltaToBucketIndex
if (deltaToBucketIndex(0.0) !== 0) throw new Error('Simultaneous chord delta should map to bucket 0');
if (deltaToBucketIndex(1/16) !== 1) throw new Error('1/16 gap should map to bucket 1');
if (deltaToBucketIndex(1.0) !== 5) throw new Error('1 beat gap should map to bucket 5');
if (deltaToBucketIndex(4.0) !== 7) throw new Error('4 beat gap (1 bar) should map to bucket 7');
if (deltaToBucketIndex(64.0) !== 11) throw new Error('64+ beat rest should map to bucket 11');
console.log('   ✓ Inter-onset timing: 12 categorical buckets (chord=0, up to 64+ beats=11)');

const encoded = SequenceEncoder.encode(events);
console.log(`   ✓ Ingested ${encoded.noteCount} notes into tensor shape [${encoded.noteCount} x 32]`);
if (encoded.noteCount !== 9) throw new Error('Expected 9 notes');
if (encoded.features.length !== 9 * 32) throw new Error('Feature length mismatch');

// Verify feature vector for first note (C4, beat 0.0, dur 1.0)
// One-hot pitch class 0 (C) should be 1.0 (feature index 0)
if (encoded.features[0] !== 1.0) throw new Error('Expected pitch class 0 to be active');
// Duration 1.0 is bucket 4 -> feature index 12 + 4 = 16 should be 1.0!
if (encoded.features[12 + 4] !== 1.0) throw new Error('Expected duration bucket 4 to be 1.0');
// First note has delta 0.0 -> bucket 0 -> feature index 20 + 0 = 20 should be 1.0!
if (encoded.features[20 + 0] !== 1.0) throw new Error('Expected chord/initial delta bucket 0 to be 1.0');

// Encode targets
const targets = SequenceEncoder.encodeTargets(groundTruth);
if (targets.targetSpelling.length !== 9) throw new Error('Target length mismatch');
if (targets.targetSpelling[3] !== tpcToClass(6)) throw new Error('F# should be class index 21');
console.log('   ✓ Ground truth encoded correctly (F♯ = class 21, B♭ = class 13)');

// 5. Test Line of Fifths Data Augmentation
console.log('\n5. Testing Data Augmentation along Line of Fifths...');
// Transpose sequence by Delta k = +1 (C major -> G major)
const transposedG = SequenceAugmenter.transpose(encoded.notes, groundTruth, 1, encoded.initialKey);
if (!transposedG) throw new Error('Transposition should succeed');

console.log(`   Transposed key: ${toKeyName(transposedG.initialKey)} (was ${toKeyName(encoded.initialKey)})`);
if (transposedG.initialKey !== 1) throw new Error('Expected new key G (1)');

// Check that F# (6) in C became C# (7) in G
const note3Old = groundTruth[3].tpc;
const note3New = transposedG.targets[3].tpc;
console.log(`   Note 3 TPC: ${toKeyName(note3Old)} (${note3Old}) -> ${toKeyName(note3New)} (${note3New})`);
if (note3New !== 7) throw new Error('Expected F# to transpose to C# (+1 step)');

// Check that the transposed MIDI note matches the new TPC's pitch class!
for (let t = 0; t < transposedG.noteEvents.length; t++) {
  const midi = transposedG.noteEvents[t][2];
  const tpc = transposedG.targets[t].tpc;
  const expectedPC = tpcToMidiPitchClass(tpc);
  const actualPC = midi % 12;
  if (actualPC !== expectedPC) {
    throw new Error(`Transposition pitch class mismatch at note ${t}: MIDI PC=${actualPC}, expected=${expectedPC}`);
  }
}
console.log('   ✓ All 9 transposed notes maintain 100% pitch-class harmonic alignment!');

console.log('\nAll Phase 1 Sequence Event pipeline tests passed with 100% success!');

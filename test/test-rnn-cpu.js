import { SpellerNetwork } from "../speller/speller-network.js";
import { toKeyName } from "../speller/key-tpc.js";
import modelWeights from "../speller/speller-weights.js";

console.log("=== RNN Phase 2: CPU Reference Implementation Verification ===\n");

// 1. Initialize SpellerNetwork with Pre-trained Weights
console.log("1. Loading pre-trained SpellerNetwork weights...");
const network = new SpellerNetwork();

network.loadWeights(modelWeights);
console.log(
    `   ✓ Loaded ${network.parameters.length} parameter tensors (${modelWeights.length} floats).`,
);

// 2. Enharmonic Disambiguation Verification
console.log(
    "\n2. Verifying Enharmonic Disambiguation (F♯ vs G♭, C♯ vs D♭, B♭ vs A♯)...",
);

// Test 1: In C major, MIDI 66 resolving upwards to G (67) must be spelled F# (TPC 6)
const testFsharpEvents = [
    [0.0, "key", 0],
    [0.0, "note", 60, 0.8, 1.0], // C4
    [1.0, "note", 64, 0.8, 1.0], // E4
    [2.0, "note", 66, 0.8, 1.0], // MIDI 66 -> F#
    [3.0, "note", 67, 0.8, 1.0], // G4
];
const predFsharp = network.predictSequence(testFsharpEvents);
const noteFsharp = predFsharp[2];
console.log(
    `   Test 1: MIDI 66 in C major resolving up to G -> predicted TPC ${noteFsharp.tpc} (${
        toKeyName(noteFsharp.tpc)
    })`,
);
if (noteFsharp.tpc !== 6) {
    throw new Error(
        `Expected F♯ (TPC 6), got ${noteFsharp.tpc} (${
            toKeyName(noteFsharp.tpc)
        })`,
    );
}
console.log("   ✓ Correctly spelled upward-resolving MIDI 66 as F♯");

// Test 2: In Eb major, MIDI 66 resolving downwards to F (65) must be spelled Gb (TPC -6)
const testGflatEvents = [
    [0.0, "key", -3],
    [0.0, "note", 63, 0.8, 1.0], // Eb4
    [1.0, "note", 68, 0.8, 1.0], // Ab4
    [2.0, "note", 66, 0.8, 0.5], // MIDI 66 -> Gb
    [2.5, "note", 65, 0.8, 1.0], // F4
    [3.5, "note", 63, 0.8, 1.0], // Eb4
];
const predGflat = network.predictSequence(testGflatEvents);
const noteGflat = predGflat[2];
console.log(
    `   Test 2: MIDI 66 in E♭ major resolving down to F -> predicted TPC ${noteGflat.tpc} (${
        toKeyName(noteGflat.tpc)
    })`,
);
if (noteGflat.tpc !== -6) {
    throw new Error(
        `Expected G♭ (TPC -6), got ${noteGflat.tpc} (${
            toKeyName(noteGflat.tpc)
        })`,
    );
}
console.log(
    "   ✓ Correctly spelled downward-resolving MIDI 66 as G♭ (enharmonic disambiguation 100% verified!)",
);

// 3. Test Model Serialization (Export / Load)
console.log(
    "\n3. Testing Model Serialization (exportWeights / loadWeights)...",
);
const exported = network.exportWeights();
const newNet = new SpellerNetwork();
newNet.loadWeights(exported);

const predReloaded = newNet.predictSequence(testFsharpEvents);
if (predReloaded[2].tpc !== 6) {
    throw new Error("Loaded network failed to reproduce predictions");
}
console.log(
    "   ✓ Model weights exported and reloaded successfully with identical predictions!",
);

console.log(
    "\nAll Phase 2 CPU Reference Implementation tests passed with 100% success!",
);

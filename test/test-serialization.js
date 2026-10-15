import deserializeWeights from "../modules/deserialize.js";
import SpellerRNN from "../speller/speller-rnn.js";
import { SpellerNetwork } from "../speller/speller-network.js";
import modelWeights from "../speller/speller-weights.js";

console.log("=== Bare Weights Serialization Round-Trip ===\n");

const bytes = await Deno.readFile(
    new URL("../speller/speller-weights.bin", import.meta.url),
);

const cpu = new SpellerNetwork();
const expected = cpu.parameters.reduce((sum, p) => sum + p.weight.length, 0);
if (bytes.byteLength !== expected * 4) {
    throw new Error(
        `.bin is ${bytes.byteLength} bytes, expected ${expected * 4}`,
    );
}
console.log(
    `   ✓ speller-weights.bin is bare: ${bytes.byteLength} bytes = ${expected} floats`,
);

// CPU loads flat Float32Array deserialized from bytes
const flatFloats = deserializeWeights(bytes);
cpu.loadWeights(flatFloats);

// GPU initializes, validates byte length, and loads in one step via SpellerRNN.from()
const gpu = await SpellerRNN.from(bytes);

if (modelWeights.byteLength !== expected * 4) {
    throw new Error("ES module modelWeights length mismatch");
}
console.log("   ✓ speller-weights.js decodes to the same float count");

// Wrong-length data must be rejected by SpellerRNN.from()
let threw = false;
try {
    await SpellerRNN.from(bytes.subarray(0, bytes.byteLength - 4));
} catch {
    threw = true;
}

if (!threw) {
    throw new Error("Truncated weights were not rejected by SpellerRNN.from()");
}
console.log(
    "   ✓ Truncated weights rejected by SpellerRNN.from() byteLength check",
);

const events = [
    [0.0, "key", 0],
    [0.0, "note", 60, 0.8, 1.0],
    [1.0, "note", 64, 0.8, 1.0],
    [2.0, "note", 66, 0.8, 0.5],
    [2.5, "note", 67, 0.8, 1.0],
    [3.5, "note", 66, 0.8, 0.5],
    [4.0, "note", 65, 0.8, 1.0],
];

const cpuPreds = cpu.predictSequence(events).map((p) => p.tpc);
const gpuPreds = (await gpu.predictSequence(events)).map((p) => p.tpc);
console.log(`   CPU: ${cpuPreds.join(", ")}`);
console.log(`   GPU: ${gpuPreds.join(", ")}`);

const target = [0, 4, 6, 1, -6, -1];
if (cpuPreds.join() !== target.join()) {
    throw new Error("CPU predictions wrong after load");
}
if (gpuPreds.join() !== target.join()) {
    throw new Error("GPU predictions wrong after load");
}

// Verify constructor directly accepts weights
const gpuFromConstructor = new SpellerRNN(gpu.device, bytes);
const gpuConstructorPreds = (await gpuFromConstructor.predictSequence(events))
    .map((p) => p.tpc);
if (gpuConstructorPreds.join() !== target.join()) {
    throw new Error("GPU constructor weight loading mismatch");
}

console.log(
    "   ✓ CPU and GPU spellers both reproduce trained predictions from bare weights",
);

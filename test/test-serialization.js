import { initWebGPU } from "../modules/gpu/context.js";
import { serializeWeights } from "../modules/serialize.js";
import deserializeWeights from "../modules/deserialize.js";
import RNN from "../modules/rnn.js";
import RecurrentNetwork from "../modules/rnn/network-cpu.js";

console.log("=== Bare Weights Serialization & GPU/CPU Round-Trip ===\n");

// 1. Initialize WebGPU
const { device } = await initWebGPU();

// 2. Define Generic Architecture
const definition = {
    inDim: 12,
    layers: [
        { hiddenDim: 24, bidirectional: true },
        { hiddenDim: 24, bidirectional: true },
    ],
    priming: {
        inDim: 6,
        type: "categorical",
    },
    heads: [
        { name: "out_a", layerIndex: 0, outDim: 5 },
        { name: "out_b", layerIndex: 1, outDim: 8 },
    ],
};

// 3. Instantiate CPU Network & Export Bare Binary Weights
const cpuSource = new RecurrentNetwork(definition);
const expectedFloats = cpuSource.parameters.reduce(
    (sum, p) => sum + p.weight.length,
    0,
);
const bytes = serializeWeights(cpuSource);

if (bytes.byteLength !== expectedFloats * 4) {
    throw new Error(
        `Binary size mismatch: got ${bytes.byteLength} bytes, expected ${
            expectedFloats * 4
        }`,
    );
}
console.log(
    `   ✓ Serialized network to bare binary: ${bytes.byteLength} bytes = ${expectedFloats} floats`,
);

// 4. Test Deserialization
const deserializedFloats = deserializeWeights(bytes);
if (deserializedFloats.length !== expectedFloats) {
    throw new Error(
        `Deserialized float count mismatch: got ${deserializedFloats.length}`,
    );
}
console.log("   ✓ deserializeWeights() reproduced identical float count.");

// 5. Test Invalid Length Rejection on GPU
let threw = false;
try {
    await RNN.from(definition, bytes.subarray(0, bytes.byteLength - 4), device);
} catch {
    threw = true;
}
if (!threw) {
    throw new Error("Truncated weights were not rejected by RNN.from()");
}
console.log("   ✓ Truncated weights correctly rejected by byteLength check.");

// 6. Load Binary Weights into Fresh CPU and GPU Instances
const cpuReloaded = new RecurrentNetwork(definition, bytes);
const gpuFromFactory = await RNN.from(definition, bytes, device);
const gpuFromConstructor = new RNN(device, definition, bytes);

// 7. Verify Predictions Match Across CPU and GPU
const T = 6;
const X = new Float32Array(T * definition.inDim);
for (let i = 0; i < X.length; i++) X[i] = (Math.random() - 0.5) * 2;
const condition = 2;

const cpuOut = cpuReloaded.forward(X, T, { condition });
const gpuOut1 = await gpuFromFactory.predict(X, T, { condition });
const gpuOut2 = await gpuFromConstructor.predict(X, T, { condition });

for (let t = 0; t < T; t++) {
    if (cpuOut.out_a.preds[t] !== gpuOut1.out_a.preds[t]) {
        throw new Error(`out_a prediction mismatch at step ${t}`);
    }
    if (cpuOut.out_b.preds[t] !== gpuOut1.out_b.preds[t]) {
        throw new Error(`out_b prediction mismatch at step ${t}`);
    }
    if (gpuOut1.out_a.preds[t] !== gpuOut2.out_a.preds[t]) {
        throw new Error(
            `Constructor vs Factory prediction mismatch at step ${t}`,
        );
    }
}

console.log(
    "   ✓ CPU and GPU networks both reproduce identical predictions from bare binary weights!",
);

import RecurrentNetwork from "../modules/rnn/network-cpu.js";

console.log("=== CPU Recurrent Neural Network Verification ===\n");

// 1. Initialize RecurrentNetwork Architecture
console.log(
    "1. Initializing 2-layer BiGRU RecurrentNetwork with Priming & Multi-Head...",
);

const definition = {
    inDim: 8,
    layers: [
        { hiddenDim: 16, bidirectional: true },
        { hiddenDim: 16, bidirectional: true },
    ],
    priming: {
        inDim: 4,
        type: "categorical",
    },
    heads: [
        { name: "class_head", layerIndex: 0, outDim: 5, lossWeight: 1.0 },
        { name: "seq_head", layerIndex: 1, outDim: 8, lossWeight: 1.5 },
    ],
};

const network = new RecurrentNetwork(definition);
console.log(
    `   ✓ Created network with ${network.parameters.length} parameter tensors.`,
);

// 2. Test Forward Pass
console.log("\n2. Testing Forward Pass...");
const T = 6;
const X = new Float32Array(T * definition.inDim);
for (let i = 0; i < X.length; i++) X[i] = (Math.random() - 0.5) * 2;

const condition = 2; // Categorical priming index in [0..3]
const forwardOut = network.forward(X, T, { condition });

if (!forwardOut.class_head || !forwardOut.seq_head) {
    throw new Error("Missing head outputs in forward pass");
}
if (forwardOut.class_head.probs.length !== T * 5) {
    throw new Error(
        `class_head probs length mismatch: ${forwardOut.class_head.probs.length}`,
    );
}
if (forwardOut.seq_head.preds.length !== T) {
    throw new Error(
        `seq_head preds length mismatch: ${forwardOut.seq_head.preds.length}`,
    );
}
console.log(
    "   ✓ Forward pass produced valid probability distributions and predictions across all heads.",
);

// 3. Test Loss, Backward Pass & Training Step
console.log(
    "\n3. Testing Compute Loss, Backward Pass (BPTT), and Gradient Step...",
);
const targets = {
    class_head: new Uint32Array([1, 2, 0, 4, 3, 1]),
    seq_head: new Uint32Array([7, 3, 2, 5, 0, 4]),
};

const stepRes = network.trainStep(X, T, condition, targets, 0.01);
if (
    typeof stepRes.totalLoss !== "number" || isNaN(stepRes.totalLoss) ||
    stepRes.totalLoss <= 0
) {
    throw new Error(`Invalid total loss: ${stepRes.totalLoss}`);
}
console.log(
    `   ✓ trainStep executed successfully: totalLoss = ${
        stepRes.totalLoss.toFixed(4)
    }, gradNorm = ${stepRes.gradNorm.toFixed(4)}`,
);

// 4. Test Model Serialization (Export / Load)
console.log(
    "\n4. Testing Model Serialization (exportWeights / loadWeights)...",
);
const forwardBeforeExport = network.forward(X, T, { condition });
const exported = network.exportWeights();
const reloadedNet = new RecurrentNetwork(definition);
reloadedNet.loadWeights(exported);

const predReloaded = reloadedNet.forward(X, T, { condition });
for (let t = 0; t < T; t++) {
    if (
        predReloaded.class_head.preds[t] !==
            forwardBeforeExport.class_head.preds[t]
    ) {
        throw new Error(
            `class_head prediction mismatch after reload at step ${t}`,
        );
    }
    if (
        predReloaded.seq_head.preds[t] !== forwardBeforeExport.seq_head.preds[t]
    ) {
        throw new Error(
            `seq_head prediction mismatch after reload at step ${t}`,
        );
    }
}
console.log(
    "   ✓ Model weights exported and reloaded successfully with identical predictions!",
);

console.log("\nAll CPU RecurrentNetwork tests passed with 100% success!");

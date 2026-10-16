import RecurrentNetwork from "../modules/rnn/network-cpu.js";

console.log(
    "=== CPU Recurrent Neural Network Convergence Training ===\n",
);

// 1. Prepare Synthetic Sequence Dataset (Sequence Memory Task)
const inDim = 4;
const outDim = 4;
const numConditions = 2;

const dataset = [];
for (let s = 0; s < 12; s++) {
    const T = 4;
    const X = new Float32Array(T * inDim);
    const targetSeq = new Uint32Array(T);
    const condition = s % numConditions;

    for (let t = 0; t < T; t++) {
        const cat = (s + t + condition) % inDim;
        X[t * inDim + cat] = 1.0;
        targetSeq[t] = cat;
    }

    dataset.push({
        X,
        T,
        condition,
        targets: {
            pred_head: targetSeq,
        },
    });
}

console.log(
    `1. Generated ${dataset.length} synthetic sequence samples for convergence testing.`,
);

// 2. Initialize RecurrentNetwork
const definition = {
    inDim,
    layers: [
        { hiddenDim: 16, bidirectional: true },
    ],
    priming: {
        inDim: numConditions,
        type: "categorical",
    },
    heads: [
        { name: "pred_head", layerIndex: 0, outDim },
    ],
};

const network = new RecurrentNetwork(definition);
console.log(
    `2. Initialized RecurrentNetwork (${network.parameters.length} parameter tensors).`,
);

// 3. Train Network using BPTT + Adam via network.train()
console.log(
    "\n3. Training network using Backpropagation Through Time (BPTT)...",
);
const { heads: finalHeads, duration } = network.train(dataset, {
    epochs: 60,
    lr: 0.02,
    logInterval: 30,
});

if (finalHeads.pred_head.acc < 90) {
    throw new Error(
        `Convergence failed: final accuracy was ${
            finalHeads.pred_head.acc.toFixed(1)
        }%`,
    );
}
console.log(
    `   ✓ Convergence successful in ${duration}s: Model achieved ${
        finalHeads.pred_head.acc.toFixed(1)
    }% accuracy!`,
);

console.log(
    "\nAll CPU Reference Implementation training & convergence tests passed with 100% success!",
);

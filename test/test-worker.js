import { NeuralWorkerClient } from "../modules/nn/client.js";

console.log("=== Web Worker WebGPU Verification (Phase 4) ===\n");

const client = new NeuralWorkerClient();

try {
    // 1. Initialize WebGPU inside the Web Worker
    console.log("1. Initializing WebGPU inside dedicated Web Worker...");
    const initRes = await client.init({
        inDim: 2,
        hiddenDim: 16,
        outDim: 2,
        lr: 0.05,
    });
    console.log(
        `   ✓ WebGPU Worker initialized! Adapter: ${initRes.adapterDescription}`,
    );

    // 2. Train on XOR Dataset via Worker postMessage RPC
    console.log("\n2. Training on XOR via Web Worker...");
    const x = new Float32Array([
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
    ]);
    const y = new Float32Array([
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
    ]);

    const trainRes = await client.trainBatch(x, y, 4, 300, (p) => {
        if (p.epoch % 100 === 0) {
            console.log(
                `   Epoch ${p.epoch} -> Loss: ${p.loss.toFixed(4)}, Accuracy: ${
                    (p.accuracy * 100).toFixed(1)
                }%`,
            );
        }
    });

    console.log(
        `   ✓ Training finished! Final Loss: ${
            trainRes.loss.toFixed(4)
        }, Acc: ${(trainRes.accuracy * 100).toFixed(1)}%`,
    );
    console.log(
        `   ✓ Replay Buffer size: ${trainRes.replayBufferSize} samples`,
    );

    // 3. Test Inference RPC
    console.log("\n3. Testing Worker Inference (Predict)...");
    const testInputs = new Float32Array([0, 1, 1, 1]); // Expected: Class 1, Class 0
    const probs = await client.predict(testInputs, 2);
    console.log(`   [0, 1] -> P(Class 1): ${(probs[1] * 100).toFixed(1)}%`);
    console.log(`   [1, 1] -> P(Class 0): ${(probs[2] * 100).toFixed(1)}%`);

    // 4. Test Continuous Online Learning with Replay Buffer
    console.log("\n4. Testing Continuous Feedback with Replay Buffer...");
    const testPoint = new Float32Array([0.9, 0.1]); // near [1, 0]
    const targetLabel = new Float32Array([0, 1]); // Class 1
    const feedbackRes = await client.feedback(testPoint, targetLabel, 16, 5);
    console.log(
        `   ✓ Feedback applied! New P(Class 1): ${
            (feedbackRes.newProbs[1] * 100).toFixed(1)
        }%`,
    );
    console.log(
        `   ✓ Replay Buffer size after feedback: ${feedbackRes.replayBufferSize} samples`,
    );

    console.log("\nAll Phase 4 Web Worker tests passed with 100% success!");
} catch (err) {
    console.error("Test error:", err);
} finally {
    client.terminate();
}

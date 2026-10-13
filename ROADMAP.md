# WebGPU Neural Network Engine: Roadmap & Architecture

An educational and high-performance neural network engine running on WebGPU inside a Web Worker, implemented in TypeScript and WGSL.

---

## 1. High-Level Architecture

```
┌────────────────────────────────────────────────────────┐
│ Main / UI Thread                                       │
│ • User interactions, live drawing/canvas, dashboard    │
│ • Human-in-the-loop feedback (correct/incorrect labels)│
└───────────────────────▲────────────────────────────────┘
                        │ postMessage()
┌───────────────────────▼────────────────────────────────┐
│ Web Worker (`worker.ts`)                               │
│ • Manages training loop, mini-batches, replay buffer   │
│ • Orchestrates WebGPU pipelines and command encoders   │
│ • Performs periodic async readbacks for telemetry      │
└───────────────────────▲────────────────────────────────┘
                        │ GPUDevice.queue.submit()
┌───────────────────────▼────────────────────────────────┐
│ WebGPU Hardware (WGSL Compute Shaders)                 │
│ 1. Forward Pass   : Tiled GEMM → Activation (ReLU/GELU)│
│ 2. Loss & Output  : Softmax + Categorical Cross-Entropy│
│ 3. Backward Pass  : Gradient GEMMs + Activation Deriv  │
│ 4. Optimizer Pass : SGD / Adam parameter update kernel │
└────────────────────────────────────────────────────────┘
```

---

## 2. Technical Stack

- **Runtime & Tooling**: Deno (pure ES modules, no `package.json`, project code in `modules/`)
- **Host Orchestration**: JavaScript (ES Modules)
- **Compute Kernels**: WGSL (WebGPU Shading Language)
- **Execution Environment**: Dedicated Web Worker (`WorkerNavigator.gpu`) / Deno WebGPU
- **Target Workload**: General classification (supports continuous / online learning)

---

## 3. Mathematical Foundations

### Forward Pass
For each layer $l \in [1, L]$ with input $A^{[l-1]}$ ($A^{[0]} = X \in \mathbb{R}^{B \times D_{\text{in}}}$):
1. **Linear Transformation**:
   $$Z^{[l]} = A^{[l-1]} W^{[l]} + \mathbf{1} \cdot (b^{[l]})^T$$
2. **Hidden Activation**:
   $$A^{[l]} = \text{ReLU}(Z^{[l]}) = \max(0, Z^{[l]})$$
3. **Output Layer (Softmax Logits)**:
   $$\hat{Y}_{i, c} = \frac{\exp(Z^{[L]}_{i, c} - \max_k Z^{[L]}_{i, k})}{\sum_k \exp(Z^{[L]}_{i, k} - \max_k Z^{[L]}_{i, k})}$$

### Backward Pass (Backpropagation)
1. **Output Gradient**:
   $$\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial Z^{[L]}} = \frac{1}{B} (\hat{Y} - Y)$$
2. **Parameter Gradients**:
   $$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = (A^{[l-1]})^T \delta^{[l]}$$
   $$\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \sum_{\text{batch}} \delta^{[l]}$$
3. **Hidden Layer Error Propagation**:
   $$\delta^{[l-1]} = (\delta^{[l]} (W^{[l]})^T) \odot \text{ReLU}'(Z^{[l-1]})$$
4. **Parameter Updates**:
   $$W^{[l]} \leftarrow W^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial W^{[l]}}$$
   $$b^{[l]} \leftarrow b^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial b^{[l]}}$$

---

## 4. Phase-by-Phase Roadmap

### Phase 1: Pure JavaScript Reference Engine (CPU)
- [x] Implement foundational linear algebra operations (`Matrix2D`, transpose, matmul, dot product, broadcast add).
- [x] Implement Dense Layer with Xavier / He weight initialization.
- [x] Implement Activation Layers (`ReLU`, `Softmax`).
- [x] Implement Loss functions (`CategoricalCrossEntropy`).
- [x] Implement full backpropagation on CPU.
- [x] Train on toy classification datasets (XOR, concentric circles, or spiral) to verify gradient correctness.

### Phase 2: WebGPU Foundation & Compute Pipeline
- [x] Set up WebGPU context and device initialization.
- [x] Implement GPU memory abstractions: buffer creation, staging uploads, GPU-to-GPU copies, async readbacks.
- [x] Write first WGSL compute shaders:
  - Element-wise operations (addition, scaling, ReLU, derivative of ReLU).
  - Tiled Matrix Multiplication (GEMM) using WGSL `workgroup` shared memory.
- [x] Validate GPU matrix operations against Phase 1 CPU reference engine.

### Phase 3: GPU Backpropagation & Optimizer
- [x] Implement full forward pass on GPU (chained GEMM + activation passes).
- [x] Implement GPU Softmax + Cross-Entropy loss gradient shader.
- [x] Implement backward pass kernels:
  - Weight gradient GEMM: $A^T \cdot \delta$
  - Input gradient GEMM: $\delta \cdot W^T$
  - Bias accumulation reduction kernel
- [x] Implement GPU-side parameter update kernels (SGD with momentum and Adam).
- [x] Maintain weights entirely in GPU `STORAGE` buffers throughout training loops.

### Phase 4: Web Worker & Continuous Learning Engine
- [x] Move the entire WebGPU device management and training loop into a Web Worker.
- [x] Implement asynchronous messaging protocol (`postMessage`) between Main Thread and Worker:
  - Inference requests (`predict`).
  - Batch training commands (`train_batch`).
  - Single feedback updates (`feedback` / `update_sample`).
- [x] Implement an **Experience Replay Buffer** to support continuous human-in-the-loop learning without catastrophic forgetting.
- [x] Build a demonstration UI (real-time canvas or interactive classifier).

---

## 5. Critical Pitfalls & Design Rules

1. **Avoid CPU-GPU Readback Stalls**: Keep weights, intermediate activations, and gradients on the GPU across training iterations. Only map buffers to the CPU when exporting models or logging loss every $N$ steps.
2. **WGSL Memory Layout & Alignment**: Respect 16-byte alignment rules for uniform buffers and structured data.
3. **Numerical Stability**: Always use numerically stable Softmax ($z - \max(z)$) and clamp probabilities to prevent $\log(0) = -\infty$.
4. **Catastrophic Forgetting in Continuous Learning**:
   If a neural network is repeatedly updated on single new samples, it tends to suffer from catastrophic forgetting: adjusting weights to satisfy the latest example can degrade its accuracy on patterns it learned previously.
   - **Experience Replay Buffer**: Keep a lightweight buffer of past samples (e.g., 200–500 examples stored in worker memory or IndexedDB). When the user gives feedback on a new sample $x_{\text{new}}$, assemble a small mini-batch: $\text{Batch} = [x_{\text{new}}] \cup [\text{random samples from replay buffer}]$. This anchors prior knowledge while adapting to new feedback.
   - **Conservative Learning Rate for Online Updates**: Use an adaptive optimizer with a moderated learning rate for user feedback steps to prevent a single noisy sample from destabilizing the network.

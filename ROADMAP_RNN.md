# WebGPU Bidirectional RNN for Musical Pitch Spelling: Architecture & Roadmap

A deep learning sequence-to-sequence model running on WebGPU inside a Web Worker, designed to solve the musical note spelling disambiguation problem (tonal pitch-class classification and local key tracking) directly on Sequence event streams.

---

## 1. Problem Formulation & Sequence Data Format

Data is supplied as a temporal array of **Sequence Event tuples/objects** (`../sequence` format):
```javascript
// Note event: [beat, type, number, gain, duration]
[0.0, "note", 60, 0.8, 1.0]

// Key event:  [beat, type, number]
[0.0, "key",  0] // 0 = C major, 1 = G major, -1 = F major, etc.
```

In written notation, MIDI pitch numbers ($p \in [0, 127]$, pitch class $p \pmod{12}$) are ambiguous (e.g., MIDI 61 = C# or Db). Correct spelling depends on:
1. **Harmonic Context**: Tertian chord construction (C-E-G, D-F-A, etc.).
2. **Local Key Tracking**: Transient tonicizations and key changes throughout the piece.
3. **Melodic Direction / Resolution**: Leading tones resolve upward (#), passing tones resolve downward (b).

Because melodic resolutions lie in the *future* and key context lies in the *past*, a **Bidirectional Recurrent Neural Network (BiGRU)** is the optimal architecture.

---

## 2. Line of Fifths Coordinate System (`soundio/sequence`)

Both **Note Spelling** and **Local Key** share the unified Line of Fifths coordinate system defined in `KEYNUMBERS`:
```javascript
// Integer coordinates centered at C = 0 (-15 to +19)
"F𝄫": -15 ... "C♭": -7 ... "F": -1, "C": 0, "G": 1 ... "C♯": 7 ... "B𝄪": 19
```

### Why Unrestricted 35 Keys for Local Key Representation?
While traditional written sheet music key signatures rarely exceed 7 accidentals ($-7 \le k \le +7$), **internal harmonic modulation frequently visits rarer regions**:
* A piece in $E$ major ($k = +4$) or $B$ major ($k = +5$) modulating to its dominant ($F\sharp$ or $C\sharp$) and tonicizing a secondary dominant enters $G\sharp$ ($k = +8$) or $D\sharp$ ($k = +9$).
* Clamping the key head to $[-7, +7]$ forces artificial discontinuities (e.g. jumping from $+7$ to $-5$), disrupting recurrent hidden states.
* Allowing the local key head to predict across all 35 keys ($-15$ to $+19$) preserves continuous voice leading along the line of fifths.
* **Notation Rendering**: When rendering the score's clef key signature, keys outside $[-7, +7]$ can be folded into the standard 15 enharmonically at presentation time.

---

## 3. Model Architecture (Two-Layer Hierarchical BiGRU)

```
Inputs per note: [Pitch Class (12), Duration One-Hot (8), Delta Beat (1)] (t = 1 ... T)
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│ Pre-projection: U = X · W_in (All time steps parallel via GEMM)│
└───────────────────────────────┬────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────┐
│ Layer 1: Bidirectional GRU (Hidden Local Key Context)          │
│ • Forward GRU (t = 0 → T-1, h_0 initialized with Initial Key)  │
│ • Backward GRU (t = T-1 → 0)                                   │
│ Output: H1_t = [h_fwd_t ; h_bwd_t]                            │
└───────────────┬────────────────────────────────┬───────────────┘
                │                                │
                ▼                                ▼
┌───────────────────────────────┐   ┌────────────────────────────┐
│ Multi-Task Head 1:            │   │ Layer 2: BiGRU             │
│ Local Key Classifier          │   │ (Tonal Pitch-Class Layer)  │
│ (35 classes: F𝄫=-15 to B𝄪=+19)│   │ Output: H2_t               │
└───────────────────────────────┘   └────────────┬───────────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────────┐
                                    │ Multi-Task Head 2:         │
                                    │ Note Spelling Classifier   │
                                    │ (35 classes: F𝄫=-15 to B𝄪) │
                                    └────────────────────────────┘
```

---

## 4. Mathematical Specifications

### A. Gated Recurrent Unit (GRU) Cell Equations
For time step $t$ with pre-projected input $u_t$ and previous hidden state $h_{t-1}$:
1. **Reset Gate**:
   $$r_t = \sigma(u_t^r + W_{hh}^r h_{t-1} + b^r)$$
2. **Update Gate**:
   $$z_t = \sigma(u_t^z + W_{hh}^z h_{t-1} + b^z)$$
3. **Candidate Hidden State**:
   $$\tilde{h}_t = \tanh(u_t^h + W_{hh}^h (r_t \odot h_{t-1}) + b^h)$$
4. **Final Hidden State**:
   $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### B. Loss Function & Tonal Centering Regularization
$$\mathcal{L} = \mathcal{L}_{\text{spelling}} + \lambda_{\text{key}} \mathcal{L}_{\text{key}} + \lambda_{\text{simplicity}} \mathcal{L}_{\text{simplicity}}$$
- Both heads evaluate 35 classes (index $= n + 15 \in [0, 34]$).
- $\lambda_{\text{key}} \approx 0.3$.
- **Accidental Simplicity Regularizer** ($\lambda_{\text{simplicity}} \approx 0.02$):
  Penalizes accidental count $|k|$ on the local key head (or log-prior bias $\text{Logit}(k) - \alpha |k|$). This pulls representations toward the center ($C = 0$) and breaks enharmonic ties in favor of fewer accidentals (e.g., preferring $D\flat$ with 5 flats over $C\sharp$ with 7 sharps).
- **Diatonic Invariant**: in local key $k$, diatonic scale tones occupy the continuous interval $[k - 1, k + 5]$.

---

## 5. Phase-by-Phase Roadmap

### Phase 1: Sequence Event Pipeline & Encodings
- [x] Import and integrate `{ KEYNAMES, KEYNUMBERS }` from `../sequence/modules/event/key.js`.
- [x] Implement `SequenceEncoder`: transforms `NoteEvent` arrays into numeric tensor batches:
  - Pitch class ($0 \dots 11$, one-hot).
  - Duration (8-bucket one-hot via soundio `roundPow2`, $1/16$ to $8$ beats).
  - Inter-onset time ($\Delta\text{beat}$).
  - Initial key conditioning ($h_0$ initialization vector).
- [x] Implement bidirectional label mappings (35 TPC classes $\leftrightarrow$ index $0 \dots 34$).
- [x] Implement Sequence-level Line of Fifths Data Augmentation:
  - Transpose both note TPCs and key signatures by integer $\Delta \in [-6, +6]$.
  - Adjust MIDI pitches by $(7 \cdot \Delta) \pmod{12}$.

### Phase 2: CPU Reference Implementation (Deno / JS)
- [x] Implement GRU Cell forward and backward pass in pure JavaScript (`modules/rnn/gru-cell.js`).
- [x] Implement Bidirectional GRU layer (`modules/rnn/bigru.js`).
- [x] Implement Generic Configurable Recurrent Network engine (`modules/rnn.js`).
- [x] Implement Pitch Speller Network with Multi-Task Heads (`speller/speller-network.js`).
- [x] Implement Backpropagation Through Time (BPTT) and Adam optimizer with gradient clipping.
- [x] Verify convergence on synthetic/canonical harmonic progressions and Line-of-Fifths augmentations (`test/test-rnn-cpu.js`).

### Phase 3: WebGPU WGSL Kernels for BiGRU
- [x] Write WGSL Pre-projection Kernel: parallel GEMM $U = X \cdot W_{\text{in}}^T + b_{\text{in}}$ across all time steps.
- [x] Write WGSL GRU Cell Compute Shader (`modules/gpu/shaders/gru.wgsl.js`):
  - Parallel workgroup vector operations across hidden dimension $H \le 256$ in on-chip SRAM (`var<workgroup>`).
  - Native WGSL activations for $\sigma(x)$ and $\tanh(x)$.
- [x] Implement bidirectional sequential dispatch (Forward $0 \to T-1$, Backward $T-1 \to 0$) in `GpuBidirectionalGru` (`modules/rnn/gpu-bigru.js`).
- [x] Implement End-to-End Generic GPU Recurrent Network (`modules/rnn/network-gpu.js`) and GPU Speller (`speller/speller-rnn.js`).
- [x] Validate GPU recurrent states and end-to-end speller predictions against Phase 2 CPU reference engine (`test/test-gpu-rnn.js`).

### Phase 4: Training & Model Serialization
- [x] Train the two-layer BiGRU on annotated sequence corpora (`speller/data/corpus.js`, `speller/train.js`).
- [x] Evaluate spelling accuracy across tonal modulations (100% accuracy on held-out enharmonic test sets).
- [x] Implement binary weight serialization (`Float32Array` export/import) with native `Uint8Array.fromBase64` / `.toBase64()` (`modules/serialize.js`, `speller/speller-weights.bin`, `speller/speller-weights.js`).
- [x] Decouple generic recurrent engine (`modules/rnn.js`) from standalone music pitch speller project (`speller/`).

### Phase 5: Client API & Interactive Demonstration
- [ ] Implement First-Class Client API taking Sequence `NoteEvent` arrays and returning annotated spelled note events (`speller/speller.js`).
- [ ] Direct main-thread WebGPU execution for low latency (no forced Web Worker).
- [ ] Build interactive demonstration (`index.html` / `app.js`) showing the sequence, predicted spellings, and local key timeline.

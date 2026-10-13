import { NeuralWorkerClient } from './modules/nn/client.js';

// Application State
const state = {
  client: null,
  adapterInfo: 'Initializing WebGPU...',
  points: [], // { x, y, label }
  selectedPoint: null,
  activeClassToAdd: 0,
  gridResolution: 40,
  isTraining: false,
  lossHistory: [],
};

// Canvas references
const canvas = document.getElementById('plotCanvas');
const ctx = canvas.getContext('2d');
const statusBadge = document.getElementById('statusBadge');
const lossValue = document.getElementById('lossValue');
const accValue = document.getElementById('accValue');
const memoryValue = document.getElementById('memoryValue');
const trainBtn = document.getElementById('trainBtn');
const feedbackPanel = document.getElementById('feedbackPanel');
const feedbackInfo = document.getElementById('feedbackInfo');
const probBar0 = document.getElementById('probBar0');
const probBar1 = document.getElementById('probBar1');

// Coordinate transforms: Normal [-1.5, 1.5] to Canvas [0, width]
function toScreen(normX, normY) {
  const w = canvas.width;
  const h = canvas.height;
  const sx = ((normX + 1.5) / 3.0) * w;
  const sy = ((1.5 - normY) / 3.0) * h;
  return { x: sx, y: sy };
}

function toNorm(screenX, screenY) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (screenX - rect.left) * scaleX;
  const y = (screenY - rect.top) * scaleY;
  const nx = (x / canvas.width) * 3.0 - 1.5;
  const ny = 1.5 - (y / canvas.height) * 3.0;
  return { x: nx, y: ny };
}

// Preset Datasets
function loadPreset(presetName) {
  state.points = [];
  if (presetName === 'xor') {
    state.points = [
      { x: -0.7, y: -0.7, label: 0 },
      { x:  0.7, y:  0.7, label: 0 },
      { x: -0.7, y:  0.7, label: 1 },
      { x:  0.7, y: -0.7, label: 1 },
      // Duplicate slightly perturbed for volume
      { x: -0.8, y: -0.6, label: 0 },
      { x:  0.6, y:  0.8, label: 0 },
      { x: -0.6, y:  0.8, label: 1 },
      { x:  0.8, y: -0.6, label: 1 },
    ];
  } else if (presetName === 'circles') {
    // Inner circle (Class 0), Outer ring (Class 1)
    const n = 16;
    for (let i = 0; i < n; i++) {
      const angle = (i / n) * Math.PI * 2;
      state.points.push({
        x: Math.cos(angle) * 0.4 + (Math.random() - 0.5) * 0.1,
        y: Math.sin(angle) * 0.4 + (Math.random() - 0.5) * 0.1,
        label: 0,
      });
      state.points.push({
        x: Math.cos(angle) * 1.0 + (Math.random() - 0.5) * 0.1,
        y: Math.sin(angle) * 1.0 + (Math.random() - 0.5) * 0.1,
        label: 1,
      });
    }
  }
  updateUI();
  render();
}

async function updateDecisionBoundary() {
  if (!state.client) return;

  const res = state.gridResolution;
  const total = res * res;
  const gridCoords = new Float32Array(total * 2);

  let idx = 0;
  for (let r = 0; r < res; r++) {
    const ny = 1.5 - (r / (res - 1)) * 3.0;
    for (let c = 0; c < res; c++) {
      const nx = (c / (res - 1)) * 3.0 - 1.5;
      gridCoords[idx * 2] = nx;
      gridCoords[idx * 2 + 1] = ny;
      idx++;
    }
  }

  const probs = await state.client.getDecisionGrid(gridCoords, total);
  render(probs);
}

function render(gridProbs = null) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 1. Draw Decision Boundary Heatmap if available
  if (gridProbs) {
    const res = state.gridResolution;
    const cellW = canvas.width / res;
    const cellH = canvas.height / res;

    for (let r = 0; r < res; r++) {
      for (let c = 0; c < res; c++) {
        const i = r * res + c;
        const p0 = gridProbs[i * 2];
        const p1 = gridProbs[i * 2 + 1];

        // Red/Orange for Class 0, Blue/Cyan for Class 1
        const rColor = Math.floor(p0 * 220);
        const bColor = Math.floor(p1 * 240);
        ctx.fillStyle = `rgba(${rColor}, 40, ${bColor}, 0.35)`;
        ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
      }
    }
  } else {
    // Draw grid lines
    ctx.strokeStyle = '#222736';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 6; i++) {
      const pos = (i / 6) * canvas.width;
      ctx.beginPath();
      ctx.moveTo(pos, 0); ctx.lineTo(pos, canvas.height);
      ctx.moveTo(0, pos); ctx.lineTo(canvas.width, pos);
      ctx.stroke();
    }
  }

  // 2. Draw Axes
  const center = toScreen(0, 0);
  ctx.strokeStyle = '#3a4259';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(center.x, 0); ctx.lineTo(center.x, canvas.height);
  ctx.moveTo(0, center.y); ctx.lineTo(canvas.width, center.y);
  ctx.stroke();

  // 3. Draw Points
  for (const pt of state.points) {
    const s = toScreen(pt.x, pt.y);
    ctx.beginPath();
    ctx.arc(s.x, s.y, 7, 0, Math.PI * 2);

    if (pt.label === 0) {
      ctx.fillStyle = '#ff4a6e'; // Class 0: Red/Pink
    } else {
      ctx.fillStyle = '#00c6ff'; // Class 1: Cyan/Blue
    }
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // 4. Highlight Selected Test Point
  if (state.selectedPoint) {
    const s = toScreen(state.selectedPoint.x, state.selectedPoint.y);
    ctx.beginPath();
    ctx.arc(s.x, s.y, 12, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffe600';
    ctx.lineWidth = 3;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

async function handleCanvasClick(e) {
  const norm = toNorm(e.clientX, e.clientY);

  if (e.shiftKey) {
    // Select point for testing/feedback
    state.selectedPoint = { x: norm.x, y: norm.y };
    render();
    await inspectSelectedPoint();
  } else {
    // Add point of active class
    state.points.push({
      x: norm.x,
      y: norm.y,
      label: state.activeClassToAdd,
    });
    updateUI();
    render();
  }
}

async function inspectSelectedPoint() {
  if (!state.selectedPoint || !state.client) return;

  const testX = new Float32Array([state.selectedPoint.x, state.selectedPoint.y]);
  const probs = await state.client.predict(testX, 1);
  const p0 = (probs[0] * 100).toFixed(1);
  const p1 = (probs[1] * 100).toFixed(1);

  feedbackPanel.classList.remove('hidden');
  probBar0.style.width = `${p0}%`;
  probBar0.textContent = `Class 0: ${p0}%`;
  probBar1.style.width = `${p1}%`;
  probBar1.textContent = `Class 1: ${p1}%`;

  const predictedClass = probs[1] > probs[0] ? 'Class 1 (Blue)' : 'Class 0 (Red)';
  feedbackInfo.textContent = `Point: (${state.selectedPoint.x.toFixed(2)}, ${state.selectedPoint.y.toFixed(2)}) → ${predictedClass}`;
}

async function applyUserFeedback(trueLabel) {
  if (!state.selectedPoint || !state.client) return;

  const targetY = new Float32Array(2);
  targetY[trueLabel] = 1.0;
  const x = new Float32Array([state.selectedPoint.x, state.selectedPoint.y]);

  // Trigger continuous online learning in Web Worker with replay buffer
  const res = await state.client.feedback(x, targetY, 16, 5);
  memoryValue.textContent = `${res.replayBufferSize} samples`;

  // Also add to visual points
  state.points.push({
    x: state.selectedPoint.x,
    y: state.selectedPoint.y,
    label: trueLabel,
  });

  await inspectSelectedPoint();
  await updateDecisionBoundary();
}

async function trainNetwork() {
  if (state.points.length === 0 || state.isTraining) return;

  state.isTraining = true;
  trainBtn.disabled = true;
  trainBtn.textContent = 'Training on WebGPU...';

  const batchSize = state.points.length;
  const x = new Float32Array(batchSize * 2);
  const y = new Float32Array(batchSize * 2);

  for (let i = 0; i < batchSize; i++) {
    x[i * 2] = state.points[i].x;
    x[i * 2 + 1] = state.points[i].y;
    y[i * 2 + state.points[i].label] = 1.0;
  }

  const epochs = parseInt(document.getElementById('epochSelect').value, 10) || 200;

  await state.client.trainBatch(x, y, batchSize, epochs, (progress) => {
    lossValue.textContent = progress.loss.toFixed(4);
    accValue.textContent = `${(progress.accuracy * 100).toFixed(1)}%`;
  });

  memoryValue.textContent = `${state.points.length} samples`;
  state.isTraining = false;
  trainBtn.disabled = false;
  trainBtn.textContent = 'Train Network (WebGPU)';

  await updateDecisionBoundary();
}

function updateUI() {
  document.getElementById('pointCount').textContent = state.points.length;
}

// Setup Event Listeners
window.addEventListener('DOMContentLoaded', async () => {
  canvas.addEventListener('click', handleCanvasClick);

  document.getElementById('btnClass0').addEventListener('click', () => {
    state.activeClassToAdd = 0;
    document.getElementById('btnClass0').classList.add('active');
    document.getElementById('btnClass1').classList.remove('active');
  });

  document.getElementById('btnClass1').addEventListener('click', () => {
    state.activeClassToAdd = 1;
    document.getElementById('btnClass1').classList.add('active');
    document.getElementById('btnClass0').classList.remove('active');
  });

  document.getElementById('btnPresetXor').addEventListener('click', () => loadPreset('xor'));
  document.getElementById('btnPresetCircles').addEventListener('click', () => loadPreset('circles'));
  document.getElementById('btnClear').addEventListener('click', async () => {
    state.points = [];
    state.selectedPoint = null;
    feedbackPanel.classList.add('hidden');
    if (state.client) await state.client.reset();
    updateUI();
    render();
  });

  trainBtn.addEventListener('click', trainNetwork);

  document.getElementById('btnFeedbackCorrect').addEventListener('click', async () => {
    const probs = await state.client.predict(new Float32Array([state.selectedPoint.x, state.selectedPoint.y]), 1);
    const predClass = probs[1] > probs[0] ? 1 : 0;
    await applyUserFeedback(predClass);
  });

  document.getElementById('btnFeedbackClass0').addEventListener('click', () => applyUserFeedback(0));
  document.getElementById('btnFeedbackClass1').addEventListener('click', () => applyUserFeedback(1));

  // Initialize Web Worker Client
  try {
    state.client = new NeuralWorkerClient();
    const initRes = await state.client.init({ inDim: 2, hiddenDim: 16, outDim: 2, lr: 0.05 });
    statusBadge.textContent = `WebGPU Active (${initRes.adapterDescription})`;
    statusBadge.classList.add('ready');

    loadPreset('xor');
    await updateDecisionBoundary();
  } catch (err) {
    statusBadge.textContent = `Error: ${err.message}`;
    statusBadge.classList.add('error');
    console.error(err);
  }
});

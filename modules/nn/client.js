/**
 * NeuralWorkerClient
 * 
 * Provides an asynchronous Promise-based API for interacting with
 * the WebGPU Neural Network Web Worker from the main thread.
 */
export class NeuralWorkerClient {
  constructor(workerPath = './worker.js') {
    this.worker = new Worker(
      new URL(workerPath, import.meta.url),
      { type: 'module' }
    );
    this._reqId = 0;
    this._pending = new Map();
    this._progressCallbacks = new Map();

    this.worker.onmessage = (event) => this._handleMessage(event.data);
    this.worker.onerror = (err) => {
      const msg = err.message || (err.filename ? `${err.message} at ${err.filename}:${err.lineno}` : 'Failed to load or execute worker script');
      console.error('NeuralWorker error:', msg, err);
    };
  }

  _send(type, payload = {}, onProgress = null) {
    const id = ++this._reqId;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      if (onProgress) {
        this._progressCallbacks.set(id, onProgress);
      }
      this.worker.postMessage({ type, id, payload });
    });
  }

  _handleMessage(data) {
    const { type, id, payload, error } = data;

    if (type === 'TRAIN_PROGRESS') {
      const cb = this._progressCallbacks.get(id);
      if (cb) cb(payload);
      return;
    }

    const pending = this._pending.get(id);
    if (!pending) return;

    if (type === 'ERROR') {
      pending.reject(new Error(error));
      this._pending.delete(id);
      this._progressCallbacks.delete(id);
      return;
    }

    pending.resolve(payload);
    this._pending.delete(id);
    this._progressCallbacks.delete(id);
  }

  async init(options = {}) {
    return await this._send('INIT', options);
  }

  async trainBatch(x, y, batchSize, epochs = 100, onProgress = null) {
    return await this._send(
      'TRAIN_BATCH',
      { x, y, batchSize, epochs },
      onProgress
    );
  }

  async predict(x, batchSize = 1) {
    const res = await this._send('PREDICT', { x, batchSize });
    return res.probs;
  }

  async feedback(x, targetY, replayBatchSize = 16, steps = 5) {
    return await this._send('FEEDBACK', {
      x,
      targetY,
      replayBatchSize,
      steps,
    });
  }

  async getDecisionGrid(gridPoints, batchSize) {
    const res = await this._send('DECISION_GRID', { gridPoints, batchSize });
    return res.probs;
  }

  async reset() {
    return await this._send('RESET');
  }

  terminate() {
    this.worker.terminate();
  }
}

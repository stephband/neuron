/**
 * Experience Replay Buffer for Continuous Online Learning
 * 
 * Stores past (input, target) experiences to prevent Catastrophic Forgetting
 * when adapting to live human-in-the-loop feedback.
 */
export class ReplayBuffer {
  constructor(featureDim, numClasses, maxCapacity = 500) {
    this.featureDim = featureDim;
    this.numClasses = numClasses;
    this.maxCapacity = maxCapacity;

    this.samplesX = [];
    this.samplesY = [];
  }

  get size() {
    return this.samplesX.length;
  }

  /**
   * Adds a single sample or batch of samples to the buffer
   * x: Float32Array of length [N * featureDim]
   * y: Float32Array of length [N * numClasses]
   */
  add(x, y, count = 1) {
    for (let i = 0; i < count; i++) {
      const sampleX = x.slice(i * this.featureDim, (i + 1) * this.featureDim);
      const sampleY = y.slice(i * this.numClasses, (i + 1) * this.numClasses);

      if (this.size >= this.maxCapacity) {
        // Evict a random historical sample to maintain diversity
        const evictIdx = Math.floor(Math.random() * this.size);
        this.samplesX[evictIdx] = sampleX;
        this.samplesY[evictIdx] = sampleY;
      } else {
        this.samplesX.push(sampleX);
        this.samplesY.push(sampleY);
      }
    }
  }

  /**
   * Samples a random mini-batch of given size
   */
  sample(batchSize) {
    const k = Math.min(batchSize, this.size);
    if (k === 0) return null;

    const batchX = new Float32Array(k * this.featureDim);
    const batchY = new Float32Array(k * this.numClasses);

    for (let i = 0; i < k; i++) {
      const idx = Math.floor(Math.random() * this.size);
      batchX.set(this.samplesX[idx], i * this.featureDim);
      batchY.set(this.samplesY[idx], i * this.numClasses);
    }

    return { x: batchX, y: batchY, batchSize: k };
  }

  /**
   * Prepares an online feedback mini-batch:
   * 1. Stores the new feedback sample into memory.
   * 2. Returns a batch containing the new sample PLUS random historical samples.
   */
  sampleWithFeedback(newX, newY, batchSize = 16) {
    this.add(newX, newY, 1);

    const k = Math.min(batchSize, this.size);
    const batchX = new Float32Array(k * this.featureDim);
    const batchY = new Float32Array(k * this.numClasses);

    // First slot is always the new sample that received feedback
    batchX.set(newX, 0);
    batchY.set(newY, 0);

    // Remaining slots drawn from replay buffer
    for (let i = 1; i < k; i++) {
      const idx = Math.floor(Math.random() * this.size);
      batchX.set(this.samplesX[idx], i * this.featureDim);
      batchY.set(this.samplesY[idx], i * this.numClasses);
    }

    return { x: batchX, y: batchY, batchSize: k };
  }

  /**
   * Clear all stored memories
   */
  clear() {
    this.samplesX = [];
    this.samplesY = [];
  }
}

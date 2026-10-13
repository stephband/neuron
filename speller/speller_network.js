import { RecurrentNetwork } from '../modules/rnn.js';
import { NUM_CLASSES, keyToClass, classToKey } from './key_tpc.js';
import { SequenceEncoder, FEATURE_DIM } from './sequence_encoder.js';

/**
 * SpellerNetwork: Musical Pitch Speller & Local Key Tracker
 * 
 * Specialized configuration of RecurrentNetwork for Sequence streams:
 * - 2-layer BiGRU (hiddenDim per direction)
 * - Priming state: conditioned on initial key signature along the Line of Fifths (35 classes)
 * - Multi-task Head 1: Local key classification with accidental simplicity prior
 * - Multi-task Head 2: Tonal pitch-class (TPC) spelling classification
 */
export class SpellerNetwork extends RecurrentNetwork {
  /**
   * @param {Object} options
   * @param {number} options.inDim - Input feature dimension (default: 32)
   * @param {number} options.hiddenDim - Hidden dimension H per direction (default: 48)
   * @param {number} options.numClasses - Number of classes for TPC & Key (default: 35)
   * @param {number} options.keyWeight - Multi-task weight for key loss lambda_key (default: 0.3)
   * @param {number} options.simplicityAlpha - Accidental simplicity prior coefficient (default: 0.05)
   */
  constructor({
    inDim = FEATURE_DIM,
    hiddenDim = 48,
    numClasses = NUM_CLASSES,
    keyWeight = 0.3,
    simplicityAlpha = 0.05,
  } = {}) {
    // Accidental simplicity penalty prior: -alpha * |k|
    const simplicityPrior = new Float32Array(numClasses);
    for (let c = 0; c < numClasses; c++) {
      simplicityPrior[c] = -simplicityAlpha * Math.abs(classToKey(c));
    }

    super({
      inDim,
      layers: [
        { hiddenDim, bidirectional: true },
        { hiddenDim, bidirectional: true },
      ],
      priming: {
        inDim: numClasses,
        type: 'categorical',
      },
      heads: [
        {
          name: 'key',
          layerIndex: 0,
          outDim: numClasses,
          lossWeight: keyWeight,
          priorBias: simplicityPrior,
        },
        {
          name: 'spelling',
          layerIndex: 1,
          outDim: numClasses,
          lossWeight: 1.0,
        },
      ],
    });

    this.hiddenDim = hiddenDim;
    this.numClasses = numClasses;
    this.keyWeight = keyWeight;
    this.simplicityAlpha = simplicityAlpha;
  }

  /**
   * Forward pass across sequence of length T
   * 
   * @param {Float32Array} X - Input features [T x inDim]
   * @param {number} T - Sequence length
   * @param {number} initialKey - Line-of-Fifths coordinate (-15 to 19)
   */
  forward(X, T, initialKey = 0) {
    const c_init = keyToClass(initialKey);
    const results = super.forward(X, T, { condition: c_init });

    return {
      probSpelling: results.spelling.probs,
      probKey: results.key.probs,
      predSpelling: results.spelling.preds,
      predKey: results.key.preds,
      logitsSpelling: results.spelling.logits,
      logitsKey: results.key.logits,
    };
  }

  /**
   * Computes multi-task cross-entropy losses and accuracies
   * 
   * @param {Object} targets - { targetSpelling, targetKey }
   */
  computeLoss(targets) {
    const namedTargets = {
      spelling: targets.targetSpelling,
      key: targets.targetKey,
    };
    const res = super.computeLoss(namedTargets);

    return {
      totalLoss: res.totalLoss,
      lossSpelling: res.heads.spelling?.loss ?? 0,
      lossKey: res.heads.key?.loss ?? 0,
      accSpelling: res.heads.spelling?.acc ?? 0,
      accKey: res.heads.key?.acc ?? 0,
    };
  }

  /**
   * Backward pass
   * 
   * @param {Object} targets - { targetSpelling, targetKey }
   */
  backward(targets) {
    const namedTargets = {
      spelling: targets.targetSpelling,
      key: targets.targetKey,
    };
    return super.backward(namedTargets);
  }

  /**
   * Complete training step: forward, compute loss, backward, clip gradients, Adam update
   */
  trainStep(X, T, initialKey, targets, lr = 0.005) {
    this.zeroGrad();
    const forwardResults = this.forward(X, T, initialKey);
    const lossResults = this.computeLoss(targets);
    this.backward(targets);
    const gradNorm = this.clipGradients(5.0);
    this.adamStep(lr);

    return {
      ...lossResults,
      gradNorm,
      ...forwardResults,
    };
  }

  /**
   * High-level inference on a Sequence event array
   * 
   * @param {Array} events - Sequence events array
   * @param {Object} options - Optional overrides (e.g. initialKey)
   * @returns {Array} Array of decoded note objects with predicted TPC coordinates and names
   */
  predictSequence(events, options = {}) {
    const encoded = SequenceEncoder.encode(events, options);
    const forwardOut = this.forward(encoded.features, encoded.noteCount, encoded.initialKey);
    const decoded = SequenceEncoder.decodePredictions(forwardOut.predSpelling, forwardOut.predKey);

    return decoded.map((d, i) => ({
      note: encoded.notes[i],
      ...d,
    }));
  }
}

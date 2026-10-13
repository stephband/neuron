import { GpuRecurrentNetwork } from '../modules/gpu/recurrent_network.js';
import { NUM_CLASSES, keyToClass, classToKey } from './key_tpc.js';
import { SequenceEncoder, FEATURE_DIM } from './sequence_encoder.js';

/**
 * GpuSpellerNetwork: WebGPU-accelerated Musical Pitch Speller & Key Tracker
 */
export class GpuSpellerNetwork extends GpuRecurrentNetwork {
  constructor(device, gpuOps, {
    inDim = FEATURE_DIM,
    hiddenDim = 48,
    numClasses = NUM_CLASSES,
    keyWeight = 0.3,
    simplicityAlpha = 0.05,
  } = {}) {
    const simplicityPrior = new Float32Array(numClasses);
    for (let c = 0; c < numClasses; c++) {
      simplicityPrior[c] = -simplicityAlpha * Math.abs(classToKey(c));
    }

    super(device, gpuOps, {
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
          priorBias: simplicityPrior,
        },
        {
          name: 'spelling',
          layerIndex: 1,
          outDim: numClasses,
        },
      ],
    });

    this.hiddenDim = hiddenDim;
    this.numClasses = numClasses;
  }

  /**
   * Complete GPU sequence inference on a Sequence event stream
   * 
   * @param {Array} events - Sequence events array
   * @param {Object} options - Optional overrides
   * @returns {Promise<Array>} Decoded array of notes with predicted TPCs and keys
   */
  async predictSequence(events, options = {}) {
    const encoded = SequenceEncoder.encode(events, options);
    const c_init = keyToClass(encoded.initialKey);

    const outputs = await this.predict(encoded.features, encoded.noteCount, { condition: c_init });
    const decoded = SequenceEncoder.decodePredictions(
      outputs.spelling.preds,
      outputs.key.preds
    );

    return decoded.map((d, i) => ({
      note: encoded.notes[i],
      ...d,
    }));
  }
}

/**
 * WebGPU Context Initializer
 *
 * Works seamlessly across:
 * - Deno CLI runtime
 * - Browser Main Thread
 * - Browser Web Worker (`WorkerNavigator.gpu`)
 */
export async function initWebGPU(options = {}) {
    const gpu = typeof navigator !== 'undefined' ? navigator.gpu : null;
    if (!gpu) throw new Error('WebGPU navigator.gpu not supported');

    const adapter = await gpu.requestAdapter({
        powerPreference: options.powerPreference || 'high-performance',
    });

    if (!adapter) throw new Error('Failed to obtain WebGPU adapter (no compatible GPU found).');

    const device = await adapter.requestDevice();

    // Handle device loss gracefully
    device.lost.then((info) => {
        console.error(`WebGPU device was lost: ${info.reason} - ${info.message}`);
    });

    return {
        device,
        adapter
    };
}

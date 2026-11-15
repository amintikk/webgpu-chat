// How the GPU's cost of one 4-bit matrix multiplication grows with the number of tokens processed at once (M).
// Gemma 4 12B's largest layer (3840 → 30720), random weights, both kernel families ONNX Runtime has
// (accuracy_level 0 and 4: int8 DP4A from M = 4). Decides whether checking several draft tokens in one pass
// (speculative decoding) can be cheaper than generating them one by one. Nothing leaves the browser.
import * as ort from 'onnxruntime-web/webgpu';

const K = 3840;
const N = 30720;
const MS = [1, 2, 3, 4, 5, 6, 8, 12, 16, 32];

export async function matmulBench(log: (line: string) => void): Promise<void> {
  const rows: string[] = [];
  log('medir multiplicaciones: inicio');
  for (const acc of [0, 4]) {
    const session = await ort.InferenceSession.create(`/bench/matmul4-acc${acc}.onnx`, {
      executionProviders: ['webgpu'],
      preferredOutputLocation: 'gpu-buffer',
    });
    const device = (await (ort.env.webgpu as unknown as { device: Promise<GPUDevice> }).device) as GPUDevice;
    const gpu = (bytes: number, fill: (u: Uint8Array<ArrayBuffer>) => void) => {
      const buffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const data = new Uint8Array(new ArrayBuffer(bytes));
      fill(data);
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const bBuf = gpu(N * (K / 32) * 16, (u) => { for (let i = 0; i < u.length; i++) u[i] = Math.imul(i, 2654435761) >>> 24; });
    const sBuf = gpu(N * (K / 32) * 2, (u) => { for (let i = 0; i < u.length; i += 2) { u[i] = 0x00; u[i + 1] = 0x20; } });
    const B = ort.Tensor.fromGpuBuffer(bBuf, { dataType: 'uint8', dims: [N, K / 32, 16] });
    const S = ort.Tensor.fromGpuBuffer(sBuf, { dataType: 'float16', dims: [N * (K / 32)] });
    const cells: string[] = [];
    for (const M of MS) {
      const a = new Uint16Array(M * K).map((_, i) => 0x3000 + (i % 97));
      const A = new ort.Tensor('float16', a, [1, M, K]);
      const run = async (n: number) => {
        for (let i = 0; i < n; i++) {
          const out = await session.run({ A, B, S });
          out.Y!.dispose();
        }
        await device.queue.onSubmittedWorkDone();
      };
      await run(3);
      const t0 = performance.now();
      await run(12);
      const ms = (performance.now() - t0) / 12;
      cells.push(`M=${M}: ${ms.toFixed(2)} ms`);
      if (M === 1) log(`medir multiplicaciones: acc ${acc} M=1 ${ms.toFixed(2)} ms`);
    }
    rows.push(`accuracy_level ${acc}: ${cells.join(' · ')}`);
    log(`medir multiplicaciones: ${rows.at(-1)}`);
    B.dispose();
    S.dispose();
    bBuf.destroy();
    sBuf.destroy();
    await session.release();
  }
  log(`MATMUL 4 bits ${K}→${N} (tiempo por pasada, incluye lanzar desde JS):\n${rows.join('\n')}`);
}

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GpuProfile } from '../src/engine/profiling.ts';
test('GPU profile converts nanoseconds, ignores invalid samples and ranks cumulative cost', () => {
  const p = new GpuProfile();
  p.add({ kernelType: 'MatMul', startTime: 0, endTime: 2e6 });
  p.add({ kernelType: 'MatMul', startTime: 2e6, endTime: 5e6 });
  p.add({ kernelType: 'Other', startTime: 0, endTime: 1e6 });
  p.add({ startTime: 10, endTime: 0 });
  p.add({ startTime: NaN, endTime: 1 });
  const s = p.summary();
  assert.equal(s.samples, 3);
  assert.equal(s.summedKernelMs, 6);
  assert.equal(s.topKernels[0]!.ms, 5);
  assert.equal(s.topKernels[0]!.calls, 2);
});
test('profile storage is bounded and zero samples remain distinguishable', () => {
  const p = new GpuProfile();
  assert.equal(p.summary().samples, 0);
  for (let i = 0; i < 10000; i++) p.add({ programName: String(i), startTime: 0, endTime: 1e6 });
  assert.equal(p.summary().samples, 10000);
  assert.equal(p.summary().summedKernelMs, 10000);
  assert.equal(p.summary().topKernels.length, 12);
});

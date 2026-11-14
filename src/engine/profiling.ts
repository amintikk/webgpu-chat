/** Bounded aggregation of ORT timestamp queries (nanoseconds), never tensor contents. */
export class GpuProfile {
  private kernels = new Map<string, { calls: number; ms: number }>();
  add(data: { kernelType?: string; programName?: string; startTime: number; endTime: number }) {
    const ms = (data.endTime - data.startTime) / 1e6;
    if (!Number.isFinite(ms) || ms < 0) return;
    let name = `${data.kernelType ?? 'unknown'}/${data.programName ?? 'unknown'}`;
    if (!this.kernels.has(name) && this.kernels.size >= 128) name = 'other';
    const entry = this.kernels.get(name) ?? { calls: 0, ms: 0 };
    entry.calls++;
    entry.ms += ms;
    this.kernels.set(name, entry);
  }
  summary() {
    const rows = [...this.kernels].map(([name, value]) => ({ name, ...value })).sort((a, b) => b.ms - a.ms);
    return { samples: rows.reduce((n, r) => n + r.calls, 0), summedKernelMs: rows.reduce((n, r) => n + r.ms, 0), topKernels: rows.slice(0, 12) };
  }
}

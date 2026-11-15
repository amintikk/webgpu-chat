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

/**
 * Summary of ONNX Runtime's own profile (session option enableProfiling), read line by line from its output:
 * time per operator and execution provider (nodes left on the CPU show up here), GPU time per kernel (with
 * timestamp-query) and the duration of each model run (one per prompt or generated token).
 */
export class OrtProfile {
  private ops = new Map<string, { calls: number; us: number }>();
  private kernels = new Map<string, { calls: number; us: number }>();
  private cpuNodes = new Set<string>();
  private runs: number[] = [];
  events = 0;

  /** One output line; true if it belonged to the profile. */
  line(s: string): boolean {
    const t = s.trim();
    if (t === '[' || t === ']') return true;
    if (!t.startsWith('{"cat"')) return false;
    let e: { cat?: string; name?: string; dur?: number; args?: Record<string, unknown> };
    try {
      e = JSON.parse(t.replace(/,\s*$/, ''));
    } catch {
      return true;
    }
    this.events++;
    const dur = Number(e.dur) || 0;
    const name = String(e.name ?? '');
    if (e.cat === 'Node' && name.endsWith('_kernel_time')) {
      const provider = String(e.args?.provider ?? '?').replace('ExecutionProvider', '');
      const op = String(e.args?.op_name ?? '?');
      add(this.ops, `${provider} ${op}`, dur);
      if (!provider.startsWith('WebGpu') && this.cpuNodes.size < 40) this.cpuNodes.add(`${op} ${name.replace(/_kernel_time$/, '')}`);
    } else if (e.cat === 'Api') {
      // GPU kernels: layer numbers removed so the 48 layers add up into one row per kind of kernel.
      add(this.kernels, name.replace(/\d+/g, '#'), dur);
    } else if (e.cat === 'Session' && name === 'model_run') {
      this.runs.push(dur);
    }
    return true;
  }

  summary(label: string): string[] {
    const ms = (us: number) => (us / 1000).toFixed(1);
    const top = (m: Map<string, { calls: number; us: number }>, n: number) =>
      [...m].sort((a, b) => b[1].us - a[1].us).slice(0, n).map(([k, v]) => `  ${ms(v.us)} ms · ${v.calls}× · ${k}`);
    const gpuTotal = [...this.kernels.values()].reduce((a, v) => a + v.us, 0);
    return [
      `PERFIL ${label}: ${this.events} eventos · ${this.runs.length} pasadas`,
      `pasadas (ms, la 1.ª es el calentamiento): ${this.runs.slice(0, 80).map((d) => Math.round(d / 1000)).join(' ')}`,
      `operadores por proveedor (tiempo de CPU al lanzar cada nodo):`,
      ...top(this.ops, 20),
      this.cpuNodes.size ? `nodos fuera de la GPU: ${[...this.cpuNodes].join(' | ')}` : 'nodos fuera de la GPU: ninguno',
      this.kernels.size ? `kernels de GPU (timestamp-query, total ${ms(gpuTotal)} ms):` : 'kernels de GPU: sin datos (timestamp-query no disponible)',
      ...top(this.kernels, 25),
    ];
  }
}

function add(m: Map<string, { calls: number; us: number }>, key: string, us: number) {
  const v = m.get(key) ?? { calls: 0, us: 0 };
  v.calls++;
  v.us += us;
  m.set(key, v);
}

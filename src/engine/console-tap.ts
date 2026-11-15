// ONNX Runtime (WebAssembly) prints its profile to standard output, which ends in console.log of this worker.
// Imported before anything else so the runtime binds this wrapper: while a sink is set, the lines it accepts
// are handed to it instead of being printed.
const print = console.log.bind(console);
let sink: ((line: string) => boolean) | null = null;
console.log = (...args: unknown[]) => {
  if (sink && args.length === 1 && typeof args[0] === 'string' && sink(args[0])) return;
  print(...args);
};

export function tapConsole(fn: ((line: string) => boolean) | null): void {
  sink = fn;
}

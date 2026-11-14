# Batería de prompts: resultados

Generado por `tools/battery/grade.mjs`. Respuestas completas en `results/battery/<modelo>.json`; notas manuales en `manual.json`.
Decodificación voraz, razonamiento desactivado, un prompt por conversación, mismos archivos ONNX (q4f16) que la web, ejecutados en CPU.

| Modelo | Conversación | Código | Razonamiento | Total | Pendientes | tok/s (CPU, orientativo) |
|---|---|---|---|---|---|---|
| lfm3-1.7b | 8/10 | 7/10 | 7/10 | 22/30 | 0 | 2.7 |

## Detalle

| Prompt | lfm3-1.7b |
|---|---|
| chat-es-1 | 0.5 |
| chat-es-2 | 1 |
| chat-es-3 | 1 |
| chat-es-4 | 0 (missing /a ver/) |
| chat-es-5 | 1 |
| chat-en-1 | 1 |
| chat-en-2 | 1 |
| chat-en-3 | 0.5 |
| chat-en-4 | 1 |
| chat-en-5 | 1 |
| code-es-1 | 1 |
| code-es-2 | 0 (Node.js v24.13.0) |
| code-es-3 | 0 (NameError: name 'contar_vocales' is not defined) |
| code-es-4 | 1 |
| code-es-5 | 1 |
| code-en-1 | 1 |
| code-en-2 | 1 |
| code-en-3 | 1 |
| code-en-4 | 1 |
| code-en-5 | 0 (/^[0-9]{5}$/) |
| reason-es-1 | 1 |
| reason-es-2 | 1 |
| reason-es-3 | 1 |
| reason-es-4 | 1 |
| reason-es-5 | 0 (missing /\b9\b/) |
| reason-en-1 | 1 |
| reason-en-2 | 1 |
| reason-en-3 | 1 |
| reason-en-4 | 0 (none of the expected answers) |
| reason-en-5 | 0 (none of the expected answers) |

# Gemma 4 12B → ONNX q4f16 (Transformers.js / WebGPU)

No había una conversión ONNX fiable de `google/gemma-4-12B-it`, así que este directorio la hace desde los
pesos oficiales y la valida contra el modelado oficial de PyTorch.

| Archivo | Qué hace |
|---|---|
| `common.py` | Lectura de tensores del safetensors oficial por HTTP Range (sin descargar los 24 GB), cuantización a 4 bits |
| `export.py` | Construye el grafo ONNX nodo a nodo y escribe los pesos en archivos de < 2 GB |
| `verify.py` | Paridad con decodificación voraz frente a PyTorch (`transformers` 5.23) |
| `MODEL_CARD.md` | Ficha del repositorio de Hugging Face |

```bash
python3 -m venv venv && ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install transformers==5.23.0 safetensors onnx onnxruntime numpy huggingface_hub
./venv/bin/python tools/gemma4/export.py --out gemma-4-12B-it-ONNX            # ~40 min, ~7,1 GB, RAM < 5 GB
./venv/bin/python tools/gemma4/verify.py --model gemma-4-12B-it-ONNX --report results/gemma4/parity.json --prepack
```

`verify.py` sin `--prepack` mapea los pesos desde disco (poca RAM, muy lento); con `--prepack` ONNX Runtime
los copia a memoria (~7,6 GB) y genera ~0,4 tokens/s en 4 núcleos ARM.

## Arquitectura reproducida

`gemma4_unified`, solo texto: 48 capas (40 de ventana deslizante de 1.024 posiciones con 8 cabezas KV de
256; 8 globales con 1 cabeza KV de 512 y K = V), 16 cabezas de consulta, escala de atención 1,0, RMSNorm
`x · w` en float32 (sin `1 + w`), normas de Q/K y V (sin peso), RoPE por defecto (θ = 10⁴) en las
capas deslizantes y RoPE proporcional al 25 % (θ = 10⁶) en las globales, MLP GeGLU (tanh), cuatro normas
por capa, `layer_scalar` por capa, embedding × √3840, cabeza LM atada al embedding y *soft-capping*
final 30 (`tanh(x/30)·30`).

## Formato

- `MatMulNBits` (4 bits, bloque 32, asimétrico con puntos cero, escalas fp16), como las exportaciones
  q4f16 de onnx-community; `gate_proj` y `up_proj` fusionados. Cabeza LM en 4 trozos de 65.536 filas.
- `GatherBlockQuantized` para el embedding de entrada (grafo `embed_tokens`).
- Activaciones fp16, normas y residuo en fp32, softmax en fp32, caché KV fp16. Las capas deslizantes
  conservan solo las últimas 1.024 posiciones. Las entradas de la caché van ordenadas con las capas
  globales primero, porque Transformers.js toma la longitud de la caché de la primera.
- `config.json` declara `gemma4` / `Gemma4ForConditionalGeneration` sin torres de visión ni audio, de modo
  que `Gemma4ForCausalLM` de Transformers.js carga solo `embed_tokens` y `decoder_model_merged`.

## Validación

`verify.py` genera con el ONNX (ONNX Runtime CPU, caché KV, voraz, 48 tokens, 6 prompts en español e
inglés de conversación, código y razonamiento) y pasa prompt + continuación por las capas oficiales
de PyTorch en una sola pasada (*teacher forcing*). En cada paso, la decodificación voraz de PyTorch
habría elegido `argmax(logits)`; si coincide con el token del ONNX en todos los pasos, la salida voraz
es idéntica. Se compara con dos pistas:

- **Cuantizada**: los mismos pesos que el ONNX (cuantizados y descuantizados) en PyTorch float32. Mide
  si el grafo es correcto. Criterio: ≥ 97 % de coincidencia y toda discrepancia en un casi empate
  (margen < 1 logit).
- **Original**: los pesos bf16 oficiales. Mide cuánto cambia la cuantización. Criterio: ≥ 90 %.

Resultado en [`results/gemma4/parity.json`](../../results/gemma4/parity.json).

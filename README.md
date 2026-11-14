<img src="assets/icon.svg" width="64" height="64" alt="">

# WebGPU Chat

Chat con modelos de lenguaje abiertos que se ejecutan **enteros en el navegador** con WebGPU. La web es
estática: los modelos se descargan una vez de Hugging Face (revisión fijada) a la caché del navegador y
las conversaciones se guardan en IndexedDB. Nada de lo que escribes sale de tu equipo.

**Pruébalo:** https://llm.aminox.dev (presentación) · https://llm.aminox.dev/app/ (el chat)

> Son modelos pequeños: pueden equivocarse e inventar datos. Comprueba lo importante.

## Modelos

| Nivel | Modelo | Archivos (q4f16) | Descarga | Licencia |
|---|---|---|---|---|
| Ligero | LFM3 1.7B (Liquid AI) | [onnx-community/LFM3-1.7B-ONNX@ab9fec4](https://huggingface.co/onnx-community/LFM3-1.7B-ONNX/tree/ab9fec4c681a25bec3b48b0b97a62fc2660a2985) | 1,11 GB | [LFM Open License v1.0](licenses/LFM-Open-License-v1.0.txt) |
| Recomendado | Qwen3.5 4B (Qwen) | [onnx-community/Qwen3.5-4B-ONNX@23ac335](https://huggingface.co/onnx-community/Qwen3.5-4B-ONNX/tree/23ac3352f288351eddcdf141b32f4f66587f7e2d), solo texto | 2,82 GB | Apache 2.0 |
| Máximo | Gemma 4 12B (Google) | exportación propia: [AMINOXDEV/gemma-4-12B-it-ONNX@90e0a27](https://huggingface.co/AMINOXDEV/gemma-4-12B-it-ONNX/tree/90e0a27bf539717b6055d7e434b9faa1f0dbd9ff) ([tools/gemma4](tools/gemma4/)) | 7,54 GB | Apache 2.0 |

La selección de modelos, las fichas (qué hacen bien, límites, equipo) y las licencias están en
`src/engine/models.ts` y `src/app/modelcards.ts`. Pruebas y medidas: [results/](results/).

## Qué incluye

- Barra lateral con las conversaciones (nueva, renombrar, borrar, buscar en títulos y mensajes), guardadas en IndexedDB.
- Respuesta en streaming con parar (Esc), regenerar y copiar; Markdown propio (sin `innerHTML` con texto del modelo) con resaltado de código (highlight.js) y botón de copiar en cada bloque.
- Selector de modelo con ficha, estado (descargado, cargado) y advertencias de compatibilidad, sin restringir la selección.
- Panel de ajustes: prompt de sistema, temperatura, top-p, máximo de tokens, razonamiento sí/no y uso del contexto. Si la conversación no cabe, se omiten los mensajes más antiguos.
- Descarga con progreso real (bytes) y gestión de modelos guardados (tamaño en caché, borrar, limpiar revisiones antiguas).
- Tema claro y oscuro, español e inglés, barra lateral como cajón en móvil.
- Adjuntar archivos (clip, arrastrar o pegar): PDF (texto extraído con pdf.js, cargado solo al adjuntar uno) y archivos de texto o código, hasta 5 por mensaje. Se leen en el navegador y se añaden al mensaje como texto; si no caben en el contexto del modelo, la app lo dice antes de enviar (`src/app/attach.ts`).
- Instalable como app (botón en la barra superior cuando el navegador lo permite; en iPhone/iPad explica cómo añadirla desde Safari).
- Todos los modelos se pueden seleccionar también en móvil. Se recomienda el ligero y se advierte del riesgo de falta de memoria con los mayores. Si una carga no termina (la pestaña se cierra por falta de memoria), al volver no se repite sola y la app explica qué ha pasado.
- Sin conexión: tras la primera visita y la descarga del modelo, la web y el chat funcionan sin internet. Un *service worker* generado en el build (`dist/sw.js`, plugin `offline` de `vite.config.ts`) guarda las páginas, el código y el runtime WebAssembly; los modelos quedan en la caché de Transformers.js. El worker fija también las peticiones de Transformers.js a la rama `main` (comprobación del tokenizador) a la revisión del modelo y las sirve desde la caché.

## Cómo funciona

- **Motor** (`src/engine/worker.ts`): Transformers.js 4.3.2 en un Web Worker, `device: 'webgpu'`, plantillas de chat oficiales de cada modelo (con `enable_thinking`).
- **Caché entre turnos**: se conserva la caché de claves y valores (y los estados recurrentes de LFM3/Qwen3.5) de la última respuesta y una instantánea al final del historial. Como las plantillas reescriben respuestas anteriores (p. ej. quitan el razonamiento), la instantánea permite reutilizar todo el historial y procesar solo la última respuesta y el mensaje nuevo. Las estadísticas de cada respuesta muestran cuántos tokens se reutilizaron.
- **Detección** (`src/engine/gpu.ts`): WebGPU, `shader-f16`, `maxBufferSize`/`maxStorageBufferBindingSize`, memoria del equipo y móvil (advertencias, no bloqueos por modelo). Sin WebGPU o sin f16 la app explica por qué y qué hacer; no hay modo WASM.
- **Red**: la CSP solo permite conectarse a la propia web, `huggingface.co` y `us.aws.cdn.hf.co` (CDN al que redirige Hugging Face, comprobado desde varios países). El runtime WebAssembly de ONNX Runtime se sirve desde `/ort/`.

## Diagnóstico de rendimiento

Activa el modo depuración antes de cargar el modelo y copia su informe después de conversar.
El registro incluye la identidad del dispositivo de ONNX Runtime cuando el navegador la expone,
los tiempos de cada respuesta, tokens reutilizados, preparación de plantilla, instantánea del contexto,
streaming de texto y renderizado. No incluye los mensajes ni se envía a ningún servidor.
Los tiempos son de pared: la instantánea incluye ejecución del modelo; streaming y renderizado
son costes parciales, no una descomposición completa de CPU/GPU. `prefillMs` mide hasta el primer token
desde después de preparar la plantilla; `totalMs` incluye también esa preparación.

La carga de pesos y sesiones incluye descarga/lectura de caché. El heap registrado corresponde solo
al JavaScript de la página; no mide memoria total, VRAM ni ocupación GPU. El tamaño de los archivos
de pesos tampoco es una medición de memoria GPU. La nota del benchmark expresa rapidez, no utilización
del hardware: usa la segunda ejecución de 128 tokens sin muestreo ni streaming de texto. Contrástala con
las líneas `CHAT` del informe, incluidos segundos turnos con caché. Hace falta un perfil del cliente
para atribuir un cuello de botella a CPU, GPU o transferencias; estas métricas no lo certifican.

### Pruebas avanzadas (Ajustes → Modo depuración)

- **Comparar muestreo**: mismo prompt y 128 tokens por respuesta, dos ejecuciones por modo;
  se informa la segunda. Compara voraz con temperatura 0,7, top-p 0,9 y top-k 40.
  Las secuencias pueden diferir: es una comparación de generación completa, no una medida aislada
  del coste del sampler. Se vacía la caché conversacional antes de empezar.
- **Perfilar GPU**: ejecuta esa comparación con `onnx.webgpu.profiling.ondata`. Agrupa tiempos
  de kernels y conserva los 12 grupos más costosos, sin tensores ni texto. Incluye calentamiento.
  El registro puede ser parcial porque los timestamps llegan asíncronamente. Sin muestras no
  significa coste cero: la función puede no estar disponible. La instrumentación altera el rendimiento;
  compara velocidades con el perfil apagado. La suma de kernels no es utilización GPU ni VRAM.
- **Contexto por bloques**: experimental, desactivado por defecto. Antes del primer uso por modelo
  cargado compara ocho tokens voraces con y sin bloques en un prompt fijo. Si difieren o falla,
  mantiene el camino normal. Si pasa, prepara las instantáneas de contexto en bloques de 128 tokens,
  conservando los estados recurrentes y liberando los intermedios. Esta prueba corta no demuestra
  paridad universal ni ahorro de memoria en todo hardware. El informe indica si se aplicó y el tiempo
  de validación; `totalMs` incluye ese tiempo, `prefillMs` no. Desmarca para volver al camino normal.
- **Parar prueba** cancela la generación en curso; puede esperar a que termine la operación actual.

Estas funciones mantienen Transformers.js/ONNX y las exportaciones fijadas. No implementan el sampler
GPU de MLC, no cambian el motor ni activan captura de grafos con formas dinámicas. Las ideas de
[WebLLM](https://github.com/mlc-ai/web-llm/blob/main/src/llm_chat.ts) se aplican aquí como pruebas
comparables y prefill experimental; el [perfil de ONNX](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html)
permite investigar el grafo antes de sustituir operaciones.

Prueba de integración CPU con pesos reales ya cacheados:
`node tools/e2e/worker-turns.mjs lfm3-1.7b q4f16 --chunk-smoke`
(y `qwen3.5-4b`). Comprueba validación local, prompt largo y continuación. Las cifras CPU no
representan el rendimiento WebGPU del cliente.

## Gemma 4 12B: exportación propia

No había un ONNX fiable de `google/gemma-4-12B-it`, así que `tools/gemma4/export.py` construye el grafo
del decodificador de texto nodo a nodo (sin cargar el modelo en memoria) leyendo los tensores del
safetensors oficial por HTTP Range, y cuantiza a 4 bits (MatMulNBits, bloque 32, escalas fp16, como las
exportaciones oficiales de onnx-community). Archivos de menos de 2 GB. `tools/gemma4/verify.py` comprueba
la paridad con el código oficial de PyTorch (`Gemma4UnifiedTextDecoderLayer`) con decodificación voraz:
detalle y resultado en [tools/gemma4/README.md](tools/gemma4/README.md).

## Desarrollo

```bash
npm install
npm run dev        # http://localhost:5173/app/  (añade ?q4 para probar en GPUs sin shader-f16; solo en desarrollo)
npm test           # i18n, Markdown, razonamiento
npm run build
./deploy.sh        # build + copia a ~/docker/webgpu-chat/data/html
```

Batería de prompts: `node tools/battery/run.mjs <modelo>` y `node tools/battery/grade.mjs <modelos…>`.

## Estructura del sitio

| Ruta | Archivo |
|---|---|
| `/`, `/en/` | `index.html`, `en/index.html` (presentación) |
| `/app/`, `/en/app/` | `app/index.html`, `src/app/main.ts` |
| `/aviso-legal/`, `/privacidad/`, `/terminos/`, `/licencias/` (y `/en/legal-notice/`, `/en/privacy/`, `/en/terms/`, `/en/licenses/`) | páginas estáticas |

Cabecera y pie compartidos en `src/site/partials/`; textos de la app con `t('…')` (`src/i18n.ts`, inglés en `src/i18n.en.ts`).
`deploy/nginx.conf` es la configuración usada en producción (CSP incluida).

## Licencia

Código bajo licencia [MIT](LICENSE). Los modelos mantienen sus licencias (LFM Open License v1.0 para
LFM3; Apache 2.0 para Qwen3.5 y Gemma 4). Transformers.js (Apache 2.0), ONNX Runtime Web (MIT),
highlight.js (BSD-3), pdf.js (Apache 2.0), Inter (OFL 1.1) y la geometría de los iconos de Lucide (ISC) mantienen las suyas;
detalle en [/licencias/](https://llm.aminox.dev/licencias/).

# Copias de los modelos en AMINOXDEV

La web descarga los modelos solo de repositorios de la cuenta AMINOXDEV en Hugging Face, para no depender
de que terceros mantengan los suyos. `mirror.py` copia, byte a byte, los archivos que usa la app desde sus
fuentes (revisiones fijadas), con un README de atribución y la licencia; `verify.py` compara cada archivo
copiado con su fuente (SHA-256 de LFS o id de blob).

```bash
python3 -m venv venv && venv/bin/pip install "huggingface_hub[hf_xet]"
venv/bin/python mirror.py lfm3|qwen|gemma   # necesita un token con escritura en ~/.cache/huggingface/token
venv/bin/python verify.py                    # actualiza antes las revisiones de `revs`
```

Tras una copia nueva: poner `repo`/`revision` en `src/engine/models.ts` y la revisión anterior en `legacy`,
para que el navegador traslade lo ya descargado en lugar de bajarlo otra vez (`src/engine/cachemove.ts`).

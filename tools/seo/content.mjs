// Content of the guide and model pages (Spanish and English). Facts come from the project itself:
// src/engine/models.ts (sizes, licences, context), results/ (measurements, parity) and the model cards.
// tools/seo/build-pages.mjs turns this into static HTML pages.

export const PAGES = [
  {
    id: 'offline',
    es: {
      path: '/ia-sin-internet/',
      title: 'Chat de IA sin internet, gratis y en tu navegador',
      metaTitle: 'IA sin internet: chat de IA offline y gratis en el navegador · WebGPU Chat',
      description: 'Habla con una IA sin conexión a internet: descarga el modelo una vez y chatea offline desde el navegador, gratis, sin instalar nada y sin que tus mensajes salgan de tu equipo.',
      crumb: 'IA sin internet',
      lead: 'La mayoría de chats de IA necesitan internet porque el modelo vive en un servidor. Aquí el modelo vive en tu equipo: se descarga una vez y, a partir de ahí, puedes hablar con la IA sin conexión.',
      sections: [
        {
          h: 'Cómo usar la IA sin conexión',
          ol: [
            'Abre <a href="/app/">el chat</a> con internet y elige un modelo. La web te recomienda el que mejor encaja con tu equipo.',
            'Pulsa «Descargar». El modelo se guarda en el almacenamiento de tu navegador (entre 1,1 y 6,8 GB, según el modelo).',
            'Ya está. La próxima vez puedes abrir la web sin conexión: la página, el código y el modelo están guardados en tu navegador.',
          ],
        },
        {
          h: 'Qué funciona sin internet',
          ul: [
            'Escribir y recibir respuestas, con el mismo modelo y la misma velocidad que con conexión.',
            'Tus conversaciones anteriores, que se guardan en tu navegador.',
            'Ajustes como el prompt de sistema, la temperatura o el razonamiento.',
          ],
          p: 'Solo necesitas conexión para abrir la web la primera vez y para descargar un modelo nuevo. Cuando no hay conexión, la app lo indica con un aviso discreto y sigue funcionando.',
        },
        {
          h: 'Por qué es posible',
          p: 'El chat usa <strong>WebGPU</strong>, la tecnología que permite a una web usar la tarjeta gráfica. Con ella, modelos de lenguaje pequeños y abiertos (LFM3, Qwen3.5 y Gemma 4) se ejecutan directamente en tu equipo con <a href="/como-funciona/">Transformers.js y ONNX Runtime Web</a>. No hay ningún servidor de IA detrás: por eso no hace falta internet, y por eso tus mensajes no salen de tu equipo.',
        },
        {
          h: 'Qué necesitas',
          ul: [
            'Un navegador con WebGPU: Chrome o Edge recientes en Windows, macOS o ChromeOS, Chrome en Android, o Safari 26.',
            'Una GPU que admita números de 16 bits (shader-f16). El modelo ligero funciona con gráficas integradas.',
            'Espacio libre para el modelo: 1,1 GB (LFM3), 2,8 GB (Qwen3.5 4B) o 6,8 GB (Gemma 4 12B).',
          ],
        },
      ],
      cta: { text: 'Abrir el chat', href: '/app/' },
      faq: [
        ['¿Es de verdad gratis?', 'Sí. No hay registro, publicidad ni suscripción: el cálculo lo hace tu equipo, así que no hay servidores que pagar.'],
        ['¿Puedo usarlo en el avión o sin cobertura?', 'Sí, si has abierto la web y descargado el modelo antes. Después funciona igual sin conexión.'],
        ['¿Es tan bueno como ChatGPT?', 'No. Son modelos pequeños que caben en un ordenador personal: útiles para redactar, resumir, traducir o programar cosas sencillas, pero pueden equivocarse más que los grandes modelos comerciales.'],
      ],
      related: ['private', 'qwen', 'how'],
    },
    en: {
      path: '/en/offline-ai-chat/',
      title: 'Offline AI chat, free and in your browser',
      metaTitle: 'Offline AI chat: free AI with no internet, in your browser · WebGPU Chat',
      description: 'Talk to an AI with no internet connection: download the model once and chat offline in your browser, for free, with nothing to install and your messages never leaving your device.',
      crumb: 'Offline AI chat',
      lead: 'Most AI chats need the internet because the model lives on a server. Here the model lives on your device: it downloads once and from then on you can talk to the AI offline.',
      sections: [
        {
          h: 'How to use the AI offline',
          ol: [
            'Open <a href="/en/app/">the chat</a> while online and pick a model. The site recommends the one that best fits your device.',
            'Press “Download”. The model is stored in your browser (1.1 to 6.8 GB depending on the model).',
            'That’s it. Next time you can open the site with no connection: the page, the code and the model are stored in your browser.',
          ],
        },
        {
          h: 'What works offline',
          ul: [
            'Writing and getting answers, with the same model and speed as online.',
            'Your previous conversations, which are saved in your browser.',
            'Settings such as the system prompt, temperature or reasoning.',
          ],
          p: 'You only need a connection to open the site the first time and to download a new model. When you are offline, the app shows a quiet note and keeps working.',
        },
        {
          h: 'Why it is possible',
          p: 'The chat uses <strong>WebGPU</strong>, the technology that lets a website use the graphics card. With it, small open language models (LFM3, Qwen3.5 and Gemma 4) run directly on your device with <a href="/en/how-it-works/">Transformers.js and ONNX Runtime Web</a>. There is no AI server behind it: that is why no internet is needed, and why your messages never leave your device.',
        },
        {
          h: 'What you need',
          ul: [
            'A browser with WebGPU: a recent Chrome or Edge on Windows, macOS or ChromeOS, Chrome on Android, or Safari 26.',
            'A GPU that supports 16-bit numbers (shader-f16). The light model runs on integrated graphics.',
            'Free space for the model: 1.1 GB (LFM3), 2.8 GB (Qwen3.5 4B) or 6.8 GB (Gemma 4 12B).',
          ],
        },
      ],
      cta: { text: 'Open the chat', href: '/en/app/' },
      faq: [
        ['Is it really free?', 'Yes. No sign-up, ads or subscription: your device does the computing, so there are no servers to pay for.'],
        ['Can I use it on a plane or with no signal?', 'Yes, if you opened the site and downloaded the model beforehand. After that it works the same offline.'],
        ['Is it as good as ChatGPT?', 'No. These are small models that fit on a personal computer: useful for writing, summarising, translating or simple coding, but they make more mistakes than large commercial models.'],
      ],
      related: ['private', 'qwen', 'how'],
    },
  },
  {
    id: 'private',
    es: {
      path: '/alternativa-privada-chatgpt/',
      title: 'Una alternativa privada a ChatGPT que funciona en tu equipo',
      metaTitle: 'Alternativa privada y gratis a ChatGPT, sin registro · WebGPU Chat',
      description: 'Chat de IA privado, gratis y sin registro: los modelos abiertos se ejecutan en tu navegador y tus conversaciones nunca salen de tu equipo. Sin cuenta, sin cookies y sin analítica.',
      crumb: 'Alternativa privada a ChatGPT',
      lead: 'Si quieres usar una IA sin enviar tus conversaciones a ninguna empresa, la opción más privada es que el modelo se ejecute en tu propio equipo. Eso es WebGPU Chat.',
      sections: [
        {
          h: 'Qué cambia frente a un chat en la nube',
          ul: [
            '<strong>Tus mensajes no salen del navegador.</strong> No hay servidor de IA: el modelo responde desde tu GPU.',
            '<strong>Sin cuenta ni registro.</strong> Abres la web y empiezas a escribir.',
            '<strong>Sin cookies ni analítica.</strong> La web no rastrea lo que haces.',
            '<strong>Funciona sin internet</strong> una vez descargado el modelo. <a href="/ia-sin-internet/">Cómo usarla offline</a>.',
            '<strong>Gratis y de código abierto.</strong> Puedes revisar el código en GitHub.',
          ],
        },
        {
          h: 'Cómo se garantiza la privacidad',
          p: 'La política de seguridad de la web (CSP) solo le permite conectarse a sí misma y a Hugging Face, de donde se descargan los modelos. Aunque quisiera, la página no podría enviar tus conversaciones a otro sitio. El historial se guarda en el almacenamiento de tu navegador y puedes borrarlo cuando quieras. Lo único que ve Hugging Face es tu dirección IP al descargar un modelo, como cualquier servidor web.',
        },
        {
          h: 'Lo que debes saber antes',
          ul: [
            'Son modelos pequeños (de 1,7 a 12 mil millones de parámetros): saben menos y se equivocan más que los grandes modelos comerciales. Comprueba lo importante.',
            'Necesitas un navegador con WebGPU y descargar el modelo una vez (1,1 a 6,8 GB).',
            'La velocidad depende de tu tarjeta gráfica. La app incluye una prueba de rendimiento con nota de 0 a 10.',
          ],
        },
        {
          h: 'Qué modelo elegir',
          ul: [
            '<a href="/modelos/lfm3-1-7b/">LFM3 1.7B</a>: el más rápido y ligero, para el día a día y equipos modestos.',
            '<a href="/modelos/qwen3-5-4b/">Qwen3.5 4B</a>: el equilibrio recomendado, con modo de razonamiento.',
            '<a href="/modelos/gemma-4-12b/">Gemma 4 12B</a>: el más capaz, para GPUs potentes.',
          ],
        },
      ],
      cta: { text: 'Empezar a chatear', href: '/app/' },
      faq: [
        ['¿Alguien puede leer mis conversaciones?', 'No. Se procesan en tu equipo y se guardan solo en tu navegador. Ni esta web ni Hugging Face las reciben.'],
        ['¿Necesito crear una cuenta?', 'No. No hay registro ni inicio de sesión.'],
        ['¿Qué pasa si borro los datos del navegador?', 'Se borran tus conversaciones y los modelos descargados. Tendrás que volver a descargar el modelo para seguir usándolo.'],
      ],
      related: ['offline', 'how', 'qwen'],
    },
    en: {
      path: '/en/private-chatgpt-alternative/',
      title: 'A private ChatGPT alternative that runs on your device',
      metaTitle: 'Private, free ChatGPT alternative with no sign-up · WebGPU Chat',
      description: 'A private, free AI chat with no sign-up: open models run in your browser and your conversations never leave your device. No account, no cookies, no analytics.',
      crumb: 'Private ChatGPT alternative',
      lead: 'If you want to use an AI without sending your conversations to any company, the most private option is to run the model on your own device. That is WebGPU Chat.',
      sections: [
        {
          h: 'What changes compared with a cloud chat',
          ul: [
            '<strong>Your messages never leave the browser.</strong> There is no AI server: the model answers from your GPU.',
            '<strong>No account or sign-up.</strong> Open the site and start typing.',
            '<strong>No cookies or analytics.</strong> The site does not track what you do.',
            '<strong>Works offline</strong> once the model is downloaded. <a href="/en/offline-ai-chat/">How to use it offline</a>.',
            '<strong>Free and open source.</strong> You can review the code on GitHub.',
          ],
        },
        {
          h: 'How privacy is guaranteed',
          p: 'The site’s security policy (CSP) only lets it connect to itself and to Hugging Face, where the models are downloaded from. Even if it wanted to, the page could not send your conversations anywhere else. Your history is stored in your browser and you can delete it whenever you like. The only thing Hugging Face sees is your IP address when you download a model, like any web server.',
        },
        {
          h: 'What to know first',
          ul: [
            'These are small models (1.7 to 12 billion parameters): they know less and make more mistakes than large commercial models. Check what matters.',
            'You need a browser with WebGPU and to download the model once (1.1 to 6.8 GB).',
            'Speed depends on your graphics card. The app includes a performance test with a 0-10 score.',
          ],
        },
        {
          h: 'Which model to choose',
          ul: [
            '<a href="/en/models/lfm3-1-7b/">LFM3 1.7B</a>: the fastest and lightest, for everyday use and modest machines.',
            '<a href="/en/models/qwen3-5-4b/">Qwen3.5 4B</a>: the recommended balance, with a reasoning mode.',
            '<a href="/en/models/gemma-4-12b/">Gemma 4 12B</a>: the most capable, for powerful GPUs.',
          ],
        },
      ],
      cta: { text: 'Start chatting', href: '/en/app/' },
      faq: [
        ['Can anyone read my conversations?', 'No. They are processed on your device and stored only in your browser. Neither this site nor Hugging Face receives them.'],
        ['Do I need an account?', 'No. There is no sign-up or login.'],
        ['What happens if I clear my browser data?', 'Your conversations and downloaded models are deleted. You will need to download the model again to keep using it.'],
      ],
      related: ['offline', 'how', 'qwen'],
    },
  },
  {
    id: 'how',
    es: {
      path: '/como-funciona/',
      title: 'Cómo funciona una IA dentro del navegador',
      metaTitle: 'Cómo ejecutar un LLM en el navegador con WebGPU · WebGPU Chat',
      description: 'Cómo se ejecuta un modelo de lenguaje (LLM) entero en el navegador: WebGPU, Transformers.js, ONNX Runtime Web, modelos cuantizados a 4 bits y caché entre turnos.',
      crumb: 'Cómo funciona',
      lead: 'WebGPU Chat no tiene servidor de inferencia. La web es un conjunto de archivos estáticos y tu navegador hace todo el trabajo. Así funciona por dentro.',
      sections: [
        {
          h: '1. WebGPU: la tarjeta gráfica desde la web',
          p: 'WebGPU es la API moderna de los navegadores para usar la GPU, tanto para dibujar como para calcular. Un modelo de lenguaje son sobre todo multiplicaciones de matrices, justo lo que la GPU hace bien. La app comprueba antes de nada que tu navegador tenga WebGPU y que la GPU admita números de 16 bits (shader-f16), y mira el tamaño máximo de búfer para bloquear los modelos que no caben.',
        },
        {
          h: '2. Modelos cuantizados a 4 bits en formato ONNX',
          p: 'Un modelo de 4.000 millones de parámetros ocupa 8 GB en 16 bits. Cuantizado a 4 bits (q4f16) baja a unos 2,8 GB, sin perder demasiada calidad. Los modelos se guardan en formato ONNX y se descargan directamente de Hugging Face, siempre de la misma versión fijada para que lo que ejecutas sea exactamente lo que se ha probado. El modelo máximo, Gemma 4 12B, lo exportamos nosotros desde los pesos oficiales y lo validamos frente al original en PyTorch.',
        },
        {
          h: '3. Transformers.js y ONNX Runtime Web',
          p: 'La inferencia corre en un Web Worker, para que la página no se congele, con Transformers.js v4 y ONNX Runtime Web en modo WebGPU. Cada modelo usa su plantilla de chat oficial, incluido el modo de razonamiento cuando se activa.',
        },
        {
          h: '4. Caché de claves y valores entre turnos',
          p: 'Un modelo de lenguaje recuerda lo leído en una caché de claves y valores. La app la conserva entre mensajes, así cada respuesta solo procesa lo nuevo y no toda la conversación. Las estadísticas de cada respuesta muestran cuántos tokens se han reutilizado.',
        },
        {
          h: '5. Todo se queda en tu equipo',
          p: 'Las conversaciones se guardan en IndexedDB y los modelos en la caché del navegador. Un service worker guarda también la web, así que todo funciona <a href="/ia-sin-internet/">sin internet</a> después de la primera descarga.',
        },
      ],
      cta: { text: 'Probarlo ahora', href: '/app/' },
      faq: [
        ['¿Qué navegadores tienen WebGPU?', 'Chrome y Edge desde la versión 113 en escritorio, Chrome 121 o posterior en Android y Safari 26 en macOS, iOS y iPadOS.'],
        ['¿Por qué no hay modo sin GPU?', 'En el procesador estos modelos irían a pocas palabras por segundo, demasiado lento para chatear.'],
        ['¿Dónde está el código?', 'En GitHub, con licencia MIT: github.com/amintikk/webgpu-chat.'],
      ],
      related: ['offline', 'gemma', 'private'],
    },
    en: {
      path: '/en/how-it-works/',
      title: 'How an AI runs inside your browser',
      metaTitle: 'How to run an LLM in the browser with WebGPU · WebGPU Chat',
      description: 'How a large language model (LLM) runs entirely in the browser: WebGPU, Transformers.js, ONNX Runtime Web, 4-bit quantized models and a cache kept between turns.',
      crumb: 'How it works',
      lead: 'WebGPU Chat has no inference server. The site is a set of static files and your browser does all the work. Here is how it works inside.',
      sections: [
        {
          h: '1. WebGPU: the graphics card from the web',
          p: 'WebGPU is the modern browser API for using the GPU, both for drawing and for computing. A language model is mostly matrix multiplications, exactly what a GPU is good at. Before anything else the app checks that your browser has WebGPU and that the GPU supports 16-bit numbers (shader-f16), and reads the maximum buffer size to block models that will not fit.',
        },
        {
          h: '2. 4-bit quantized models in ONNX format',
          p: 'A 4-billion-parameter model takes 8 GB at 16 bits. Quantized to 4 bits (q4f16) it drops to about 2.8 GB without losing too much quality. The models are stored in ONNX format and downloaded straight from Hugging Face, always at the same pinned version, so what you run is exactly what was tested. The largest model, Gemma 4 12B, we exported ourselves from the official weights and validated against the original in PyTorch.',
        },
        {
          h: '3. Transformers.js and ONNX Runtime Web',
          p: 'Inference runs in a Web Worker, so the page never freezes, with Transformers.js v4 and ONNX Runtime Web in WebGPU mode. Each model uses its official chat template, including the reasoning mode when turned on.',
        },
        {
          h: '4. Key-value cache kept between turns',
          p: 'A language model remembers what it has read in a key-value cache. The app keeps it between messages, so each answer only processes what is new instead of the whole conversation. Each answer’s statistics show how many tokens were reused.',
        },
        {
          h: '5. Everything stays on your device',
          p: 'Conversations are stored in IndexedDB and models in the browser cache. A service worker stores the site too, so everything works <a href="/en/offline-ai-chat/">offline</a> after the first download.',
        },
      ],
      cta: { text: 'Try it now', href: '/en/app/' },
      faq: [
        ['Which browsers have WebGPU?', 'Chrome and Edge from version 113 on desktop, Chrome 121 or later on Android, and Safari 26 on macOS, iOS and iPadOS.'],
        ['Why is there no CPU mode?', 'On the processor these models would manage a few words per second, too slow to chat.'],
        ['Where is the code?', 'On GitHub, under the MIT licence: github.com/amintikk/webgpu-chat.'],
      ],
      related: ['offline', 'gemma', 'private'],
    },
  },
  {
    id: 'lfm',
    model: 'lfm3-1.7b',
    es: {
      path: '/modelos/lfm3-1-7b/',
      title: 'LFM3 1.7B en tu navegador',
      metaTitle: 'LFM3 1.7B online y gratis en el navegador (WebGPU) · WebGPU Chat',
      description: 'Usa LFM3 1.7B de Liquid AI gratis en tu navegador, sin instalar nada y sin internet tras la descarga (1,1 GB). Qué hace bien, límites, requisitos y licencia.',
      crumb: 'LFM3 1.7B',
      lead: 'LFM3 1.7B es el modelo ligero de WebGPU Chat: un modelo de lenguaje de Liquid AI con 1.700 millones de parámetros, pensado para ser rápido en equipos modestos.',
      facts: [['Autor', 'Liquid AI'], ['Parámetros', '1.700 millones'], ['Descarga', '1,1 GB (q4f16)'], ['Contexto en la web', '8.192 tokens'], ['Idiomas', 'Español, inglés y otros 14'], ['Licencia', 'LFM Open License v1.0']],
      sections: [
        { h: 'Qué hace bien', ul: ['Respuestas rápidas incluso con gráficas integradas.', 'Reescribir, resumir y traducir textos cortos.', 'Extraer datos y responder con un formato pedido, como listas o JSON.'] },
        { h: 'Sus límites', ul: ['Sabe pocos datos del mundo: inventa nombres, fechas y cifras con facilidad.', 'Flojo en razonamiento de varios pasos y en programación; el propio fabricante no lo recomienda para eso.'], p: 'En nuestra batería de 30 preguntas en español e inglés acertó 22: 8 de 10 en conversación, 7 de 10 en código (con tests ejecutados) y 7 de 10 en razonamiento.' },
        { h: 'Qué equipo necesitas', p: 'Un navegador con WebGPU y una GPU con shader-f16, con unos 2 GB de memoria gráfica libre. Funciona bien en portátiles recientes con gráfica integrada.' },
        { h: 'Licencia', p: 'LFM Open License v1.0 de Liquid AI: uso y redistribución libres, pero el uso comercial solo está permitido a entidades con menos de 10 millones de dólares de ingresos anuales. <a href="/licencias/#lfm3">Texto completo</a>.' },
      ],
      cta: { text: 'Chatear con LFM3 1.7B', href: '/app/?model=lfm3-1.7b' },
      faq: [
        ['¿LFM3 1.7B funciona sin internet?', 'Sí. Tras descargarlo una vez (1,1 GB) puedes usarlo sin conexión.'],
        ['¿Funciona en el móvil?', 'Sí, en Chrome para Android con WebGPU y en Safari 26, si la GPU admite shader-f16. Es el modelo recomendado para móviles.'],
      ],
      related: ['qwen', 'gemma', 'offline'],
    },
    en: {
      path: '/en/models/lfm3-1-7b/',
      title: 'LFM3 1.7B in your browser',
      metaTitle: 'LFM3 1.7B online and free in your browser (WebGPU) · WebGPU Chat',
      description: 'Use Liquid AI’s LFM3 1.7B for free in your browser, with nothing to install and no internet after the download (1.1 GB). What it does well, limits, requirements and licence.',
      crumb: 'LFM3 1.7B',
      lead: 'LFM3 1.7B is WebGPU Chat’s light model: a language model by Liquid AI with 1.7 billion parameters, designed to be fast on modest machines.',
      facts: [['Author', 'Liquid AI'], ['Parameters', '1.7 billion'], ['Download', '1.1 GB (q4f16)'], ['Context on the site', '8,192 tokens'], ['Languages', 'English, Spanish and 14 more'], ['License', 'LFM Open License v1.0']],
      sections: [
        { h: 'What it does well', ul: ['Fast answers even on integrated graphics.', 'Rewriting, summarising and translating short texts.', 'Extracting data and answering in a requested format, such as lists or JSON.'] },
        { h: 'Its limits', ul: ['Knows few facts about the world: easily makes up names, dates and figures.', 'Weak at multi-step reasoning and coding; the maker itself does not recommend it for that.'], p: 'In our battery of 30 questions in Spanish and English it got 22 right: 8 of 10 in conversation, 7 of 10 in code (with executed tests) and 7 of 10 in reasoning.' },
        { h: 'Hardware you need', p: 'A browser with WebGPU and a GPU with shader-f16, with about 2 GB of free graphics memory. Runs well on recent laptops with integrated graphics.' },
        { h: 'License', p: 'Liquid AI’s LFM Open License v1.0: free use and redistribution, but commercial use is only allowed for entities with under $10 million in annual revenue. <a href="/en/licenses/#lfm3">Full text</a>.' },
      ],
      cta: { text: 'Chat with LFM3 1.7B', href: '/en/app/?model=lfm3-1.7b' },
      faq: [
        ['Does LFM3 1.7B work offline?', 'Yes. After downloading it once (1.1 GB) you can use it with no connection.'],
        ['Does it work on phones?', 'Yes, in Chrome for Android with WebGPU and in Safari 26, if the GPU supports shader-f16. It is the recommended model for phones.'],
      ],
      related: ['qwen', 'gemma', 'offline'],
    },
  },
  {
    id: 'qwen',
    model: 'qwen3.5-4b',
    es: {
      path: '/modelos/qwen3-5-4b/',
      title: 'Qwen3.5 4B en tu navegador',
      metaTitle: 'Qwen3.5 4B online y gratis en el navegador, con razonamiento · WebGPU Chat',
      description: 'Usa Qwen3.5 4B gratis en tu navegador con WebGPU, con modo de razonamiento, sin registro y sin internet tras la descarga (2,8 GB). Ejemplos reales, requisitos y licencia.',
      crumb: 'Qwen3.5 4B',
      lead: 'Qwen3.5 4B es el modelo recomendado de WebGPU Chat: el mejor equilibrio entre calidad y velocidad para la mayoría de ordenadores, con un modo de razonamiento que puedes activar con un botón.',
      facts: [['Autor', 'Qwen (Alibaba Cloud)'], ['Parámetros', '4.000 millones'], ['Descarga', '2,8 GB (q4f16)'], ['Contexto en la web', '8.192 tokens'], ['Razonamiento', 'Sí, activable'], ['Imágenes', 'Sí'], ['Licencia', 'Apache 2.0']],
      sections: [
        { h: 'Qué hace bien', ul: ['Explicaciones, redacción y programación de complejidad media.', 'Con el razonamiento activado, resuelve mejor problemas de matemáticas y lógica.', 'Buen nivel en español e inglés.', 'Ve imágenes: adjunta una foto o una captura (o pégala con Ctrl+V) y pregúntale por ella; se procesa en tu equipo, como el texto.'], p: 'En la <a href="/#ejemplos">portada</a> tienes diez respuestas reales suyas, sin editar: un haiku, la relatividad para niños, un plan de viaje, código en Python, una tabla y más.' },
        { h: 'Sus límites', ul: ['Más lento que LFM3 y con una descarga de casi 3 GB.', 'El razonamiento alarga mucho las respuestas: puede tardar minutos en equipos modestos.', 'Puede equivocarse en datos concretos; compruébalos.', 'La primera imagen que le envías descarga su lector de imágenes (672 MB).'] },
        { h: 'Qué equipo necesitas', p: 'Un navegador con WebGPU y shader-f16, unos 4 GB de memoria gráfica libre y 8 GB de RAM o más. Puedes probarlo también en móvil, aunque puede faltar memoria y cerrarse la pestaña.' },
        { h: 'Licencia', p: 'Apache 2.0 de Alibaba Cloud (Qwen): uso libre, también comercial, conservando el aviso de licencia. La conversión a ONNX es de onnx-community; la web la descarga, en una versión fijada, de una copia en nuestra cuenta de Hugging Face.' },
      ],
      cta: { text: 'Chatear con Qwen3.5 4B', href: '/app/?model=qwen3.5-4b' },
      faq: [
        ['¿Cómo activo el razonamiento de Qwen3.5?', 'Con el botón «Razonar» junto a la caja de texto, o en los ajustes. El modelo piensa antes de responder y verás su razonamiento plegado encima de la respuesta.'],
        ['¿Qwen3.5 4B funciona sin internet?', 'Sí, después de descargarlo una vez (2,8 GB).'],
      ],
      related: ['lfm', 'gemma', 'private'],
    },
    en: {
      path: '/en/models/qwen3-5-4b/',
      title: 'Qwen3.5 4B in your browser',
      metaTitle: 'Qwen3.5 4B online and free in your browser, with reasoning · WebGPU Chat',
      description: 'Use Qwen3.5 4B for free in your browser with WebGPU, with a reasoning mode, no sign-up and no internet after the download (2.8 GB). Real examples, requirements and licence.',
      crumb: 'Qwen3.5 4B',
      lead: 'Qwen3.5 4B is WebGPU Chat’s recommended model: the best balance between quality and speed for most computers, with a reasoning mode you can turn on with one button.',
      facts: [['Author', 'Qwen (Alibaba Cloud)'], ['Parameters', '4 billion'], ['Download', '2.8 GB (q4f16)'], ['Context on the site', '8,192 tokens'], ['Reasoning', 'Yes, optional'], ['Pictures', 'Yes'], ['License', 'Apache 2.0']],
      sections: [
        { h: 'What it does well', ul: ['Explanations, writing and medium-complexity coding.', 'With reasoning on, it solves maths and logic problems better.', 'Good level in English and Spanish.', 'Sees pictures: attach a photo or a screenshot (or paste it with Ctrl+V) and ask about it; it is processed on your device, like the text.'], p: 'On the <a href="/en/#ejemplos">home page</a> you will find ten of its real, unedited answers: a haiku, relativity for kids, a travel plan, Python code, a table and more.' },
        { h: 'Its limits', ul: ['Slower than LFM3, with a download of almost 3 GB.', 'Reasoning makes answers much longer: it can take minutes on modest machines.', 'It can get specific facts wrong; check them.', 'The first picture you send it downloads its picture reader (672 MB).'] },
        { h: 'Hardware you need', p: 'A browser with WebGPU and shader-f16, about 4 GB of free graphics memory and 8 GB of RAM or more. You can also try it on phones, although memory may run out and the tab may close.' },
        { h: 'License', p: 'Apache 2.0 by Alibaba Cloud (Qwen): free use, including commercial, keeping the licence notice. The ONNX conversion is by onnx-community; the site downloads it, at a pinned version, from a copy in our own Hugging Face account.' },
      ],
      cta: { text: 'Chat with Qwen3.5 4B', href: '/en/app/?model=qwen3.5-4b' },
      faq: [
        ['How do I turn on Qwen3.5 reasoning?', 'With the “Reasoning” button next to the text box, or in the settings. The model thinks before answering and you will see its reasoning folded above the answer.'],
        ['Does Qwen3.5 4B work offline?', 'Yes, after downloading it once (2.8 GB).'],
      ],
      related: ['lfm', 'gemma', 'private'],
    },
  },
  {
    id: 'gemma',
    model: 'gemma4-12b',
    es: {
      path: '/modelos/gemma-4-12b/',
      title: 'Gemma 4 12B en tu navegador',
      metaTitle: 'Gemma 4 12B en el navegador con WebGPU, gratis y offline · WebGPU Chat',
      description: 'Ejecuta Gemma 4 12B de Google en tu navegador con WebGPU: versión QAT de Google a 4 bits con atención fusionada, sin internet tras la descarga (6,8 GB). Requisitos y licencia.',
      crumb: 'Gemma 4 12B',
      lead: 'Gemma 4 12B es el modelo máximo de WebGPU Chat: el más capaz de los tres, con respuestas más completas y precisas, para equipos con una GPU potente.',
      facts: [['Autor', 'Google DeepMind'], ['Parámetros', '12.000 millones'], ['Descarga', '6,8 GB (q4f16)'], ['Contexto en la web', '4.096 tokens'], ['Idiomas', 'Más de 140'], ['Licencia', 'Apache 2.0']],
      sections: [
        { h: 'Qué hace bien', ul: ['Respuestas más completas y precisas que los modelos pequeños.', 'Redacción cuidada, resúmenes largos, programación y razonamiento.', 'Muy buen español y más de 140 idiomas.', 'Ve imágenes (en pruebas): adjunta una foto o una captura y pregúntale por ella; puede fallar más que el original porque este motor no aplica la atención en ambos sentidos que Gemma usa dentro de cada imagen.'] },
        { h: 'Pesos QAT y atención fusionada', p: 'Usamos la versión QAT de Google (gemma-4-12B-it-qat-q4_0), entrenada para funcionar a 4 bits, por lo que pierde menos calidad que una cuantización directa. La conversión a ONNX es de la comunidad (james58899/gemma-4-12B-it-qat-ONNX) y usa atención fusionada (GroupQueryAttention), que en WebGPU es mucho más rápida; nosotros solo activamos en su grafo el cálculo en enteros para leer el prompt más rápido.' },
        { h: 'Sus límites', ul: ['Descarga de 6,8 GB y carga lenta la primera vez.', 'Necesita una GPU potente: en equipos justos fallará al cargar o irá muy lento.', 'Algo menos preciso que el original por la cuantización a 4 bits.', 'Contexto limitado a 4.096 tokens en esta web para contener la memoria.'] },
        { h: 'Qué equipo necesitas', p: 'Una GPU dedicada con al menos 10 GB de memoria (o un Mac con chip M y 16 GB o más) y 16 GB de RAM. Puedes probarlo en móvil, aunque puede faltar memoria y cerrarse la pestaña.' },
        { h: 'Licencia', p: 'Apache 2.0 de Google (Gemma 4): uso libre, también comercial, conservando el aviso de licencia. La conversión mantiene la misma licencia.' },
      ],
      cta: { text: 'Chatear con Gemma 4 12B', href: '/app/?model=gemma4-12b' },
      faq: [
        ['¿Mi equipo puede ejecutar Gemma 4 12B?', 'Si tienes una GPU dedicada con 10 GB o más, o un Mac con chip M y 16 GB, probablemente sí. La app comprueba tu GPU y la prueba de rendimiento te da una nota de 0 a 10.'],
        ['¿Gemma 4 12B funciona sin internet?', 'Sí, después de descargarlo una vez (6,8 GB).'],
      ],
      related: ['qwen', 'how', 'offline'],
    },
    en: {
      path: '/en/models/gemma-4-12b/',
      title: 'Gemma 4 12B in your browser',
      metaTitle: 'Gemma 4 12B in the browser with WebGPU, free and offline · WebGPU Chat',
      description: 'Run Google’s Gemma 4 12B in your browser with WebGPU: Google’s 4-bit QAT version with fused attention, no internet after the download (6.8 GB). Requirements and licence.',
      crumb: 'Gemma 4 12B',
      lead: 'Gemma 4 12B is WebGPU Chat’s largest model: the most capable of the three, with more complete and accurate answers, for machines with a powerful GPU.',
      facts: [['Author', 'Google DeepMind'], ['Parameters', '12 billion'], ['Download', '6.8 GB (q4f16)'], ['Context on the site', '4,096 tokens'], ['Languages', 'More than 140'], ['License', 'Apache 2.0']],
      sections: [
        { h: 'What it does well', ul: ['More complete and accurate answers than the small models.', 'Careful writing, long summaries, coding and reasoning.', 'Very good Spanish and more than 140 languages.', 'Sees pictures (in testing): attach a photo or a screenshot and ask about it; it may do worse than the original because this engine does not apply the two-way attention Gemma uses inside each picture.'] },
        { h: 'QAT weights and fused attention', p: 'We use Google’s QAT version (gemma-4-12B-it-qat-q4_0), trained to run at 4 bits, so it loses less quality than plain quantization. The ONNX conversion comes from the community (james58899/gemma-4-12B-it-qat-ONNX) and uses fused attention (GroupQueryAttention), which is much faster on WebGPU; we only switch on integer math in its graph so prompts are read faster.' },
        { h: 'Its limits', ul: ['A 6.8 GB download and a slow first load.', 'Needs a powerful GPU: on modest machines it will fail to load or run very slowly.', 'Slightly less accurate than the original because of 4-bit quantization.', 'Context limited to 4,096 tokens on this site to keep memory in check.'] },
        { h: 'Hardware you need', p: 'A dedicated GPU with at least 10 GB of memory (or an Apple silicon Mac with 16 GB or more) and 16 GB of RAM. You can try it on phones, although memory may run out and the tab may close.' },
        { h: 'License', p: 'Apache 2.0 by Google (Gemma 4): free use, including commercial, keeping the licence notice. The conversion keeps the same licence.' },
      ],
      cta: { text: 'Chat with Gemma 4 12B', href: '/en/app/?model=gemma4-12b' },
      faq: [
        ['Can my computer run Gemma 4 12B?', 'If you have a dedicated GPU with 10 GB or more, or an Apple silicon Mac with 16 GB, probably yes. The app checks your GPU and the performance test gives you a 0-10 score.'],
        ['Does Gemma 4 12B work offline?', 'Yes, after downloading it once (6.8 GB).'],
      ],
      related: ['qwen', 'how', 'offline'],
    },
  },
];

// Per-model information sheets ("fichas"), shown in the app and on the landing page.
import { t } from '../i18n.ts';
import type { ModelId } from '../engine/models.ts';

export interface Card {
  good: string[];
  limits: string[];
  hardware: string;
  licenseNote: string;
}

export function card(id: ModelId): Card {
  switch (id) {
    case 'lfm3-1.7b':
      return {
        good: [
          t('Respuestas rápidas incluso con gráficas integradas.'),
          t('Conversación sencilla, reescribir, resumir y traducir textos cortos.'),
          t('Extraer datos y responder con un formato pedido (listas, JSON).'),
          t('Entiende y escribe en español, inglés y otros 14 idiomas.'),
        ],
        limits: [
          t('Sabe pocos datos del mundo: inventa nombres, fechas y cifras con facilidad.'),
          t('Flojo en razonamiento de varios pasos y en programación; el fabricante no lo recomienda para eso.'),
        ],
        hardware: t('GPU con WebGPU y shader-f16, unos 2 GB de memoria gráfica libre. Va bien en portátiles recientes.'),
        licenseNote: t(
          'Licencia LFM Open License v1.0 de Liquid AI: uso y redistribución libres, pero el uso comercial solo está permitido a entidades con menos de 10 millones de dólares de ingresos anuales.',
        ),
      };
    case 'qwen3.5-4b':
      return {
        good: [
          t('El mejor equilibrio entre calidad y velocidad de esta web.'),
          t('Explicaciones, redacción y programación de complejidad media.'),
          t('Con el razonamiento activado, resuelve mejor problemas de matemáticas y lógica.'),
          t('Buen nivel en español e inglés.'),
        ],
        limits: [
          t('Más lento que LFM3 y con una descarga de casi 3 GB.'),
          t('El razonamiento alarga mucho las respuestas: puede tardar minutos en equipos modestos.'),
          t('Puede equivocarse en datos concretos; compruébalos.'),
          t('Aquí solo trabaja con texto (el modelo original también admite imágenes).'),
        ],
        hardware: t('GPU con WebGPU y shader-f16, unos 4 GB de memoria gráfica libre y 8 GB de RAM o más. En móviles puede faltar memoria.'),
        licenseNote: t('Licencia Apache 2.0 de Alibaba Cloud (Qwen): uso libre, también comercial, conservando el aviso de licencia.'),
      };
    case 'gemma4-12b':
      return {
        good: [
          t('El modelo más capaz de la web: respuestas más completas y precisas.'),
          t('Redacción cuidada, resúmenes largos, programación y razonamiento.'),
          t('Muy buen español y más de 140 idiomas.'),
        ],
        limits: [
          t('Descarga de 6,8 GB y carga lenta la primera vez.'),
          t('Necesita una GPU potente; en equipos justos fallará al cargar o irá muy lento.'),
          t('Exportación propia del modelo oficial cuantizada a 4 bits: algo menos precisa que el original.'),
          t('Contexto limitado a 4.096 tokens en esta web para contener la memoria.'),
        ],
        hardware: t('GPU dedicada con al menos 10 GB de memoria (o un Mac con chip M y 16 GB o más) y 16 GB de RAM.'),
        licenseNote: t('Licencia Apache 2.0 de Google (Gemma 4): uso libre, también comercial, conservando el aviso de licencia.'),
      };
  }
}

export function tierLabel(tier: 'light' | 'recommended' | 'max'): string {
  return tier === 'light' ? t('Ligero') : tier === 'recommended' ? t('Recomendado') : t('Máximo');
}

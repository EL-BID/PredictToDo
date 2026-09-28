# Preparación para Code4Dev

Revisión local: 28 de septiembre de 2026. Referencia: indicaciones de Jesenia del
16 de septiembre y autorización de la SDM del 24 de septiembre, reenviada por
Camila el 25. Esta revisión no certifica la publicación. Se comprobó acceso de lectura al S3
del proyecto y se inspeccionó el paquete del modelo central.

## Cambios documentales preparados

- Nombre público: PredictToDo.
- README sin badges ni afirmación de inclusión ya completada en el catálogo.
- Inventario de fuentes contrastado con `src/prepare_dataset.py`.
- Alcance de autorización: únicamente `datos_procesados`; excluye datos crudos.
- Clima y población identificados como entradas del pipeline. Falta comprobar
  su presencia en el orden de variables del checkpoint seleccionado.

## Pendientes para cerrar la entrega técnica

| Entregable | Evidencia local | Criterio de cierre |
|---|---|---|
| Fuentes documentadas | El README contiene inventario con campos pendientes | Completar proveedor, enlace, corte y condiciones usando el linaje de los artefactos |
| Datos de ejemplo | Procesados localizados en S3; sin muestra local de inferencia | Seleccionar zona y 1–2 semanas de `datos_procesados`; verificar autorización, agregación e identificadores |
| Grafo y artefactos | Localizados en S3; pendientes de recortar y alinear | Incluir grafo, estáticos, seeds, mapa regional, normalización y orden de variables compatibles |
| Pesos del modelo | Modelos regionales localizados en S3; paquete central inspeccionado | Elegir checkpoint, confirmar alcance de autorización y comunicar a Jesenia si se publican ahora |
| Notebook autónomo | Los notebooks de trabajo requieren datos externos | Ejecutar de principio a fin en un entorno limpio, con rutas relativas o URLs de Releases y sin AWS ni consultas OSM en vivo |
| Distribución | Carpeta de contexto retirada del árbol versionado e ignorada | Preparar una distribución sin contexto interno y revisar archivos, salidas de notebooks y referencias internas restantes |
| Comunicación | Camila pidió fecha de entrega el 22/9 | Enviar enlaces verificables a cambios, ejemplo probado y artefactos; indicar bloqueos restantes |

## Contrato del ejemplo

Ubicar el ejemplo en `examples/bogota/`. Solo los datos procesados revisados van
en `examples/bogota/data/`; conservar excluidos los directorios operativos y los
datos crudos. Usar la misma versión de fuentes que en el entrenamiento.

Antes de ejecutar inferencia, comprobar:

1. Correspondencia de IDs de vías entre grafo, datos, seeds y mapa regional.
2. Orden y dimensiones de variables compatibles con el checkpoint.
3. Normalización y configuración procedentes del mismo modelo.
4. Ventanas temporales suficientes para entrada y horizonte de salida.
5. Registro de versión, tamaño y SHA-256 para cada artefacto distribuido.
6. Ejecución completa sin credenciales, descargas de portales en vivo ni rutas
   de servidores internos. Mostrar predicciones del modelo entrenado y explicar
   sus límites; no sustituirlo por entrenamiento sobre una muestra pequeña.

Según la solicitud de Jesenia, los archivos de más de 50 MB van en Releases;
para pesos de más de 2 GB hay que acordar otra opción con su equipo.

## Exclusión de contexto interno

La carpeta de contexto de trabajo se conserva únicamente como contenido local
ignorado por Git. Se retiró del árbol versionado y se excluye de los archivos
exportados. También se retiraron las referencias de la documentación pública.
Esta eliminación no reescribe los commits históricos del repositorio.

## Pasos posteriores del equipo Code4Dev

Tras recibir los entregables, Jesenia y su equipo prevén transferir el
repositorio, ejecutar SonarQube y GitGuardian, revisar documentación y licencia,
preparar la vista del catálogo y solicitar la autorización formal de publicación.
Ya existe `LICENSE.md` localmente; corresponde al equipo BID validar que sea la
versión aplicable. Ningún cambio local implica aprobación de publicación.

## Inventario S3 comprobado

Se confirmó lectura del prefijo de proyecto `congestion-prediction/co/movilidadbogota/`.
Las rutas completas y el paquete descargado se conservan en el área local ignorada
`data/code4dev-review/`, fuera de los archivos preparados para publicación.

| Artefacto | Resultado de inspección |
|---|---|
| `datos_procesados/conectivity_graph/road_graph.pickle` | Localizado; 186.873.872 bytes; no descargado ni recortado |
| `datos_procesados/sparse_datasets/static_dataset.h5` | Localizado; 71.434.760 bytes; pendiente de alinear al subgrafo |
| `datos_procesados/sparse_datasets/normalization_constants.pkl` | Localizado; 7.897 bytes; compatibilidad pendiente de verificar |
| `datos_procesados/sparse_datasets/jams_variables_dataset.h5` | Localizado; 34.143.067.720 bytes; requiere extracción acotada |
| Seeds regionales y manifiesto | Localizados en `artifacts/regional_seed_lists/` |
| Modelo GConvGRU central del 15/05/2026 | Descargado para inspección local; 242.842 bytes; SHA-256 coincide con manifiesto |
| Modelo GConvGRU no central del 15/05/2026 | Localizado; 242.035 bytes; no ejecutado |

El paquete central contiene `model.pth`, `model_config.pkl`, `data_meta.pkl`,
`training_args.pkl`, `metrics.pkl` y `feature_order.json`. Se inspeccionaron los
nombres de sus entradas y el JSON sin deserializar los archivos pickle.
El contrato declara 130 variables dinámicas totales, 11 futuras y 4 meteorológicas.
Esto confirma la necesidad de documentar clima para este candidato. El linaje
exacto de población en las variables estáticas sigue pendiente de revisar.

El manifiesto describe estos modelos como candidatos finales para despliegue y
pruebas, entrenados con artefactos regionales parciales. No se ha probado inferencia
ni seleccionado formalmente uno para la distribución pública. Los pesos son
pequeños: el volumen de los datasets, no el tamaño del modelo, es el trabajo
principal de preparación del ejemplo.

# Muestra preprocesada del centro de Bogotá

Datos para ejecutar [el notebook local](../../notebooks/local_training_example.ipynb)
desde un clon del repositorio, sin descargar las fuentes completas. El paquete
incluye 21 archivos (45.746.325 bytes): grafo, diez HDF5 y sus diez metadatos.

- Periodo: 1 de octubre de 2024, 00:00, a 15 de octubre, 23:45; 1.440 intervalos.
  Los timestamps no incluyen zona horaria; su zona no está confirmada.
- Selección: semilla `17000085_N.1` (CL 12A), próxima a (4.600, −74.075),
  y 4.999 nodos descubiertos mediante BFS dirigido. Se conservan las aristas
  originales entre esos nodos, con sus atributos.
- Grafo: 5.000 vías y 21.986 aristas, fuertemente conexo. Incluye 10.908 aristas
  de influencia con `legal=0`: conectividad del modelo no equivale a rutas legales
  de conducción. La selección no corresponde a límites administrativos.
- Extensión aproximada: 6,8 km este-oeste y 5,9 km norte-sur. Coordenadas de extremos
  en EPSG:3116 según el código de construcción; límites en `manifest.json`.
- Objetivo: `mean_jam_level`. Hay filas de congestión en 1.407 intervalos; no implica
  que todas las vías estén observadas en cada instante.

## Archivos y ejecución

`data/graph.pkl` contiene el grafo. `static_dataset.h5` contiene las variables
estáticas; `jams_dataset.h5`, los objetivos. Las ocho entradas dinámicas son
`coi_points`, `coi_tracts`, `events`, `incidents`, `irregularities`, `jam_delay`,
`jam_level_count` y `jam_velocity`, cada una con sufijo `_dataset.h5`.
Cada HDF5 tiene su propio archivo `_metadata.pkl`. Se usan las fuentes separadas,
no el agregado canónico `jams_variables_dataset`.

Abrir el notebook y ejecutar las celdas en orden con Python 3.12 y las dependencias
del proyecto. Comprueba los hashes SHA-256 de `manifest.json` antes de cargar los
pickle. Usa por defecto 32 vías, 12 pasos históricos, 4 futuros, dos épocas y
hasta 16 ventanas por partición. Los 5.000 nodos están disponibles para ampliar
el ejemplo; no se procesan todos simultáneamente en la configuración por defecto.
Los resultados se guardan localmente en `data/example_output/`.

## Ausencias y alcance

No se ha establecido si cada ausencia dispersa significa falta de observación,
valor cero omitido o ausencia de evento. El notebook conserva el relleno a cero
del cargador para las entradas, como simplificación computacional. Para el objetivo
construye una máscara a partir de las filas explícitas de `mean_jam_level`, incluidos
los ceros explícitos: solo esas posiciones intervienen en pérdida, MAE y referencia
constante. Esta política no recupera observaciones ausentes ni elimina el sesgo de
cobertura; las métricas describen únicamente los registros presentes seleccionados.

Se separan periodos cronológicamente (60/20/20), sin ventanas que crucen sus límites.
La normalización se ajusta solo con las ventanas de entrenamiento. No se incluyen
pesos preentrenados, clima ni normalización anual. No se afirma representatividad
geográfica o temporal, compatibilidad con pesos publicados ni calidad predictiva.

## Procedencia y reproducción

La muestra procede de los datos preprocesados del proyecto: primero se recortaron
los conjuntos temporales ordenados al periodo indicado, reindexando tiempos a
0..1439; después se seleccionaron las vías y reindexaron los nodos. Se conserva
la precisión y se usa compresión HDF5 gzip nivel 4 con shuffle.

`extract_sample.py` reproduce el segundo paso a partir de una carpeta de ciudad
ya recortada a esos 15 días, con grafo y metadatos alineados:

```bash
python examples/bogota/extract_sample.py /ruta/city /ruta/nueva_muestra
```

La carpeta destino debe ser nueva. La receta comprueba conectividad fuerte,
mapas compartidos, orden temporal y valores finitos. No descarga ni reconstruye
las fuentes originales. `manifest.json` registra tamaños, filas, cobertura y hashes
de la versión incluida. La licencia de software del repositorio no establece por
sí sola una licencia independiente para las fuentes de datos; esta ficha no les
atribuye condiciones de uso adicionales.

## Comprobación de ejecución

El 29/09/2026 se ejecutaron todas las celdas en CPU, limitando PyTorch a dos hilos
para la comprobación y manteniendo los parámetros del notebook. Pasaron carga,
entrenamiento de dos épocas, evaluación, guardado y equivalencia al recargar pesos.
Se seleccionaron 16 ventanas de entrenamiento, 12 de validación y 11 de prueba;
estas últimas contenían 132 objetivos explícitos. El MAE de prueba fue 1,033 frente
a 0,460 de la referencia constante. Esta ejecución verifica el procedimiento;
el entrenamiento corto no supera esa referencia ni acredita calidad predictiva.

*Esta herramienta digital forma parte del catálogo de herramientas del **Banco Interamericano de Desarrollo**. Puedes conocer más sobre la iniciativa del BID en [Código para el Desarrollo](https://knowledge.iadb.org/es/codigo-para-el-desarrollo).*

<h1 align="center">Bogotá Traffic Congestion Forecasting</h1>

<p align="center">
Sistema de preparación de datos, entrenamiento, evaluación y generación de artefactos para predicción de congestión vehicular en Bogotá con redes neuronales gráficas espacio-temporales.
</p>

## Tabla de contenidos

- [Badges o escudos](#badges-o-escudos)
- [Descripción y contexto](#descripción-y-contexto)
- [Qué contiene este repositorio](#qué-contiene-este-repositorio)
- [Cómo funciona el sistema](#cómo-funciona-el-sistema)
- [Guía de usuario](#guía-de-usuario)
- [Guía de instalación](#guía-de-instalación)
- [Dependencias](#dependencias)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Datos y artefactos](#datos-y-artefactos)
- [Entrenamiento](#entrenamiento)
- [Inferencia y evaluación](#inferencia-y-evaluación)
- [Calidad y validación](#calidad-y-validación)
- [Cómo contribuir](#cómo-contribuir)
- [Código de conducta](#código-de-conducta)
- [Autores](#autores)
- [Información adicional](#información-adicional)
- [Licencia](#licencia)
- [Limitación de responsabilidades](#limitación-de-responsabilidades)
- [Acknowledgments / Reconocimientos](#acknowledgments--reconocimientos)

## Badges o escudos

![Python](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-active-green)
![Tests](https://img.shields.io/badge/tests-118%20passing-brightgreen)

El badge de SonarCloud solicitado por la plantilla del BID puede agregarse cuando el proyecto tenga configurado el análisis estático público correspondiente.

## Descripción y contexto

Este repositorio contiene el código de investigación y operación para pronosticar congestión vehicular en Bogotá a partir de datos grafo-temporales. El proyecto construye datasets dispersos, entrena modelos temporales de Graph Neural Networks, reconstruye predicciones para inferencia y mantiene herramientas de evaluación offline y de evaluación densa sobre ventanas recientes.

La herramienta no es solo un pipeline de preparación de datos. Su alcance actual incluye preprocesamiento, construcción de artefactos, definición de modelos, entrenamiento en SageMaker, muestreo balanceado por severidad de congestión, calibración, evaluación y documentación técnica para mantenimiento.

El sistema representa la red vial como un grafo: los segmentos de vía son nodos y las conexiones de flujo, dirección o proximidad vial son aristas. Sobre ese grafo se combinan atributos estáticos de la vía con señales dinámicas observadas o conocidas en el tiempo. El modelo aprende tanto la evolución temporal de la congestión como su propagación espacial entre segmentos conectados.

## Qué contiene este repositorio

El repositorio incluye:

- preprocesamiento grafo-temporal disperso sobre datos de vías, Waze, PMT, eventos, incidentes, clima, paraderos, edificaciones y estado vial;
- `TrafficDataset`, un dataset de batches gráficos locales por seed usado por entrenamiento y evaluación;
- modelos temporales GNN como `GConvLSTM`, `GConvGRU` y `TGCN`;
- entrypoints de entrenamiento compatibles con AWS SageMaker;
- notebooks para lanzar entrenamientos, calibración y construcción de artefactos;
- muestreo por artefactos `(seed_id, time_idx)` para balancear exposición a niveles de congestión;
- utilidades de calibración, reconstrucción de predicciones y evaluación densa;
- documentación técnica en `docs/`;
- un white paper en LaTeX en `white_paper/`;
- una copia del README técnico original en [README_original.md](README_original.md).

## Cómo funciona el sistema

El flujo principal del sistema es:

1. Los datos fuente se almacenan en S3, EFS o carpetas locales como `data/`.
2. Los módulos de preprocesamiento generan features estáticas, dinámicas, meteorológicas, gráficas y de normalización.
3. Los artefactos se integran en datasets HDF5 dispersos alineados por nodo y tiempo.
4. `TrafficDataset` materializa muestras temporales sobre subgrafos locales definidos por seeds.
5. El entrenamiento ejecuta modelos GNN temporales y funciones de pérdida orientadas a niveles de congestión.
6. Los artefactos opcionales de seed-time balancean el entrenamiento entre buckets de congestión alta, media y baja.
7. Los scripts de evaluación y calibración producen métricas, transformaciones y evidencia para seleccionar modelos.
8. Los artefactos finales se entregan al repositorio de servicio en tiempo real.

Este repositorio corresponde al lado de entrenamiento, artefactos y evaluación. El servicio de inferencia en tiempo real vive en un repositorio separado y tiene supuestos de ejecución diferentes.

## Guía de usuario

Los usuarios principales son equipos técnicos, científicos de datos y responsables de operación que necesiten preparar datos, entrenar modelos, validar desempeño o generar artefactos para despliegue.

Flujo recomendado para nuevos usuarios técnicos:

1. Leer [docs/system-overview.md](docs/system-overview.md) para entender los límites del sistema.
2. Revisar [docs/artifacts-and-contracts.md](docs/artifacts-and-contracts.md) para conocer los contratos de datos, grafo, modelo y seeds.
3. Configurar el ambiente local siguiendo [docs/setup.md](docs/setup.md).
4. Ejecutar o inspeccionar el pipeline de datos documentado en [docs/data-pipeline.md](docs/data-pipeline.md).
5. Entrenar mediante `src/training_entrypoint.py` o los notebooks de SageMaker.
6. Evaluar resultados con los scripts y guías de [docs/inference-evaluation.md](docs/inference-evaluation.md).

Comandos principales:

```bash
# Preparar el dataset procesado
uv run python -m src.prepare_dataset

# Ejecutar el entrypoint canónico de entrenamiento
python -m src.training_entrypoint

# Ejecutar pruebas locales
make test

# Ejecutar lint
make lint

# Formatear archivos Python
make format
```

Si se cuenta con acceso AWS, los datos fuente pueden sincronizarse desde S3:

```bash
uv run -- dotenv run -- aws s3 sync \
  s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/ \
  ./data
```

## Guía de instalación

El proyecto está estandarizado en Python `3.12`.

Crear un ambiente local nuevo:

```bash
conda create -n congestion-bogota-py312 python=3.12 -y
conda activate congestion-bogota-py312
python -m pip install uv
uv sync --all-groups
```

Si los paquetes de extensión de PyTorch Geometric intentan compilarse localmente, instala ruedas precompiladas compatibles con la versión local de PyTorch. Para el ambiente CPU validado actualmente:

```bash
python -m pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.9.0+cpu.html
```

Validar la instalación:

```bash
make test
make lint
```

## Dependencias

Dependencias principales:

- Python `3.12`.
- PyTorch.
- PyTorch Geometric.
- PyTorch Geometric Temporal.
- Pandas, NumPy, SciPy, HDF5 y PyArrow.
- GeoPandas, Shapely, OSMnx y NetworkX.
- AWS SDK, AWS CLI y SageMaker.
- Ray y Dask para flujos distribuidos o procesamiento intensivo.
- Jupyter, Matplotlib, Seaborn y Plotly para análisis, notebooks y visualización.

Las dependencias completas están declaradas en [pyproject.toml](pyproject.toml) y bloqueadas en [uv.lock](uv.lock).

## Estructura del repositorio

```text
.
├── src/                 # Paquete principal: preprocessing, dataset, modelos, training e inferencia
├── tests/               # Pruebas unitarias y de comportamiento
├── notebooks/           # Lanzadores SageMaker y notebooks exploratorios
├── scripts/             # Scripts operativos, evaluación y artefactos
├── docs/                # Documentación técnica y operativa
├── ai-context/          # Notas de handoff y estado de experimentos
├── white_paper/         # Fuente LaTeX del white paper del proyecto
├── pyproject.toml       # Dependencias y configuración de herramientas
├── uv.lock              # Resolución bloqueada de dependencias
├── Makefile             # Comandos locales comunes
└── Dockerfile.sagemaker # Receta de imagen para SageMaker
```

Documentación recomendada:

- [docs/README.md](docs/README.md): índice de documentación.
- [docs/repository-structure.md](docs/repository-structure.md): mapa del código.
- [docs/process-runbooks.md](docs/process-runbooks.md): procedimientos operativos.
- [docs/code-reference.md](docs/code-reference.md): inventario de clases y funciones.
- [ai-context/00-start-here.md](ai-context/00-start-here.md): estado operativo más reciente para trabajo asistido por IA.

## Datos y artefactos

Los datos grandes y artefactos generados no se versionan en Git. Según el flujo, pueden vivir en:

- `data/`, `datos/` o `datos_procesados/` localmente;
- `/data/datos_procesados` en SageMaker/EFS;
- buckets S3 bajo `s3://sagemaker-congestion-bogota/...`;
- artefactos compartidos bajo `s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/...`.

El grafo vial es un grafo de NetworkX cuyos IDs de nodo deben coincidir con los `road_id` usados por features, targets, seeds y mapas regionales. Un path local típico es:

```text
data/datos_procesados/conectivity_graph/road_graph.pickle
```

`TrafficDataset` espera artefactos alineados por grafo y tiempo. Los tensores principales del modelo son:

| Tensor | Significado |
|---|---|
| `x_static` | features estáticas por nodo |
| `x_dynamic_past` | features dinámicas históricas en la ventana de entrada |
| `x_dynamic_future` | covariables futuras conocidas en el horizonte de predicción |
| `edge_index` | aristas del subgrafo local del seed seleccionado |
| `y` | nivel de congestión objetivo y, opcionalmente, velocidad |

El orden de features es un contrato. Cualquier cambio en generación de features debe reflejarse en los metadatos guardados y verificarse contra entrenamiento e inferencia.

## Entrenamiento

El entrypoint canónico de entrenamiento es:

```bash
python -m src.training_entrypoint
```

En la práctica, los entrenamientos se lanzan normalmente desde notebooks de SageMaker en `notebooks/`, especialmente:

```text
notebooks/sagemaker_training_job.ipynb
```

El entrenamiento soporta:

- muestreo antiguo basado en seeds;
- muestreo por artefactos `(seed_id, time_idx)`;
- familias de modelo `GConvLSTM`, `GConvGRU` y `TGCN`;
- ruta histórica o de soporte para `A3TGCN`;
- funciones de pérdida compuestas para congestión;
- componente auxiliar de forma temporal para seguir mejor la trayectoria futura;
- entrenamiento regional para corredores centrales y no centrales mediante mapas de región.

Los artefactos de modelo entrenado deben incluir pesos, configuración, orden de features, constantes de normalización o referencias a ellas, y metadatos suficientes para trazar la línea de entrenamiento.

Más detalles:

- [docs/training.md](docs/training.md)
- [docs/rebalance_sampler_design.md](docs/rebalance_sampler_design.md)

## Inferencia y evaluación

Los componentes principales de inferencia y reconstrucción son:

- `src/inference_dataloader.py`
- `src/inference_production.py`
- `src/prediction_reconstruction.py`

La inferencia difiere del entrenamiento porque lee datos fuente actuales, no targets HDF5 históricos, y reconstruye las predicciones hacia salidas por nodo y timestamp.

Los scripts de evaluación están en `scripts/` e incluyen utilidades para backfill, evaluación densa reciente, métricas por horizonte y transformaciones de calibración. La evaluación densa compara predicciones parquet con archivos de jams, declarando explícitamente cómo se interpretan valores faltantes.

Más detalles:

- [docs/inference-evaluation.md](docs/inference-evaluation.md)
- [docs/validation.md](docs/validation.md)

## Calidad y validación

Estado de calidad local más reciente documentado para Python `3.12.12`:

- `118` pruebas pasan.
- `ruff check .` pasa.

Antes de entrenar, publicar artefactos o cambiar contratos de datos, se recomienda ejecutar:

```bash
make test
make lint
```

Las tareas pendientes y riesgos de mantenimiento están documentados en [docs/cleanup-plan.md](docs/cleanup-plan.md).

## Cómo contribuir

Para contribuir:

1. Revisa la documentación en [docs/README.md](docs/README.md) y el estado actual en [ai-context/00-start-here.md](ai-context/00-start-here.md).
2. Crea una rama descriptiva para el cambio.
3. Mantén los cambios acotados al módulo correspondiente.
4. Ejecuta validaciones locales antes de abrir un pull request.

Las contribuciones deben preservar los contratos principales:

- un batch de entrenamiento usa un solo grafo local asociado a un seed;
- `edge_index` debe corresponder al conjunto local de nodos;
- el orden de features debe coincidir con el esperado por el modelo;
- los artefactos de entrenamiento, calibración e inferencia deben documentar ventana temporal, configuración y linaje.

## Código de conducta

Este repositorio no incluye todavía un archivo `CODE_OF_CONDUCT.md`. Se recomienda agregar uno siguiendo las plantillas públicas de GitHub o los lineamientos institucionales aplicables antes de abrir el proyecto a contribuciones externas amplias.

Mientras tanto, se espera que toda interacción técnica sea respetuosa, colaborativa y orientada a resolver problemas de manera transparente.

## Autores

Proyecto desarrollado en el marco de trabajo asociado al sistema de predicción de congestión para Bogotá y su documentación técnica. Antes de publicar nombres personales o correos, validar la información institucional correspondiente.

## Información adicional

El white paper técnico se encuentra en [white_paper/traffic_forecasting.tex](white_paper/traffic_forecasting.tex), con referencias en [white_paper/references.bib](white_paper/references.bib). Los archivos auxiliares generados por LaTeX no deben versionarse.

La copia del README técnico previo se conserva en [README_original.md](README_original.md) para facilitar comparación histórica y recuperar detalles operativos si se necesitan.

## Licencia

La licencia del proyecto está incluida en [LICENSE.md](LICENSE.md). Consulta ese archivo para conocer los permisos, condiciones y restricciones de uso, modificación y distribución de la herramienta.

## Limitación de responsabilidades

El BID no será responsable, bajo circunstancia alguna, de daño ni indemnización, moral o patrimonial; directo o indirecto; accesorio o especial; o por vía de consecuencia, previsto o imprevisto, que pudiese surgir:

i. Bajo cualquier teoría de responsabilidad, ya sea por contrato, infracción de derechos de propiedad intelectual, negligencia o bajo cualquier otra teoría; y/o

ii. A raíz del uso de la Herramienta Digital, incluyendo, pero sin limitación de potenciales defectos en la Herramienta Digital, o la pérdida o inexactitud de los datos de cualquier tipo. Lo anterior incluye los gastos o daños asociados a fallas de comunicación y/o fallas de funcionamiento de computadoras, vinculados con la utilización de la Herramienta Digital.

## Acknowledgments / Reconocimientos

**Copyright © [2025]. Inter-American Development Bank ("IDB"). Authorized Use.**
The procedures and results obtained based on the execution of this software are those programmed by the developers and do not necessarily reflect the views of the IDB, its Board of Executive Directors or the countries it represents.

**Copyright © [2025]. Banco Interamericano de Desarrollo ("BID"). Uso Autorizado.**
Los procedimientos y resultados obtenidos con la ejecución de este software son los programados por los desarrolladores y no reflejan necesariamente las opiniones del BID, su Directorio Ejecutivo ni los países que representa.

### Support and Usage Documentation / Documentación de Soporte y Uso

**Copyright © [2025]. Inter-American Development Bank ("IDB").** The Support and Usage Documentation is licensed under the Creative Commons License CC-BY 4.0 license. The opinions expressed in the Support and Usage Documentation are those of its authors and do not necessarily reflect the opinions of the IDB, its Board of Executive Directors, or the countries it represents.

**Copyright © [2025]. Banco Interamericano de Desarrollo (BID).** La Documentación de Soporte y Uso está licenciada bajo la licencia Creative Commons CC-BY 4.0. Las opiniones expresadas en la Documentación de Soporte y Uso son las de sus autores y no reflejan necesariamente las opiniones del BID, su Directorio Ejecutivo ni los países que representa.

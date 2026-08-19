*Esta herramienta digital forma parte del catalogo de herramientas del **Banco Interamericano de Desarrollo**. Puedes conocer mas sobre la iniciativa del BID en [Codigo para el Desarrollo](https://knowledge.iadb.org/es/codigo-para-el-desarrollo).*

<h1 align="center">Bogota Traffic Congestion Forecasting</h1>

<p align="center">
Sistema de prediccion de congestion vehicular para Bogota basado en redes neuronales graficas espacio-temporales.
</p>

## Tabla de contenidos

- [Badges o escudos](#badges-o-escudos)
- [Descripcion y contexto](#descripcion-y-contexto)
- [Guia de usuario](#guia-de-usuario)
- [Guia de instalacion](#guia-de-instalacion)
- [Dependencias](#dependencias)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Como contribuir](#como-contribuir)
- [Codigo de conducta](#codigo-de-conducta)
- [Autores](#autores)
- [Informacion adicional](#informacion-adicional)
- [Licencia](#licencia)
- [Limitacion de responsabilidades](#limitacion-de-responsabilidades)
- [Acknowledgments / Reconocimientos](#acknowledgments--reconocimientos)

## Badges o escudos

![Python](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-active-green)
![Tests](https://img.shields.io/badge/tests-118%20passing-brightgreen)

El badge de SonarCloud solicitado por la plantilla del BID puede agregarse cuando el proyecto tenga configurado el analisis estatico publico correspondiente.

## Descripcion y contexto

Este repositorio contiene el codigo de investigacion, entrenamiento, construccion de artefactos y evaluacion del sistema de prediccion de congestion vehicular de Bogota. La herramienta construye datasets espacio-temporales dispersos, entrena modelos de Graph Neural Networks temporales, reconstruye pronosticos para inferencia y permite evaluar resultados de manera offline y sobre ventanas recientes.

El sistema modela la red vial como un grafo: los segmentos de via son nodos y las relaciones de flujo o proximidad vial son aristas. A partir de informacion historica y contextual, el modelo aprende como evoluciona la congestion en el tiempo y como se propaga entre vias conectadas.

La herramienta integra fuentes como:

- datos de jams e irregularidades de trafico;
- incidentes viales;
- eventos publicos;
- planes de manejo de transito;
- clima;
- estado de la malla vial;
- paraderos de bus;
- edificaciones y atributos urbanos;
- atributos estaticos de la red vial.

El repositorio se enfoca en el lado de entrenamiento, preparacion de datos, calibracion y evaluacion. La implementacion de servicio en tiempo real se mantiene en un repositorio separado.

## Guia de usuario

Los usuarios principales de este repositorio son equipos tecnicos, cientificos de datos y responsables de operacion que necesiten preparar datos, entrenar modelos, evaluar desempeno o generar artefactos para despliegue.

Flujo general de uso:

1. Preparar o sincronizar las fuentes de datos requeridas.
2. Ejecutar el pipeline de preprocesamiento para generar artefactos grafo-temporales.
3. Entrenar el modelo con el entrypoint compatible con SageMaker.
4. Evaluar el desempeno con las utilidades offline o de evaluacion densa reciente.
5. Publicar los artefactos necesarios para el servicio de inferencia.

Comandos principales:

```bash
# Preparar el dataset procesado
uv run python -m src.prepare_dataset

# Ejecutar el entrypoint de entrenamiento
python -m src.training_entrypoint

# Ejecutar pruebas locales
make test

# Ejecutar lint
make lint
```

Para procedimientos operativos detallados, consulta:

- [docs/README.md](docs/README.md): indice de documentacion.
- [docs/system-overview.md](docs/system-overview.md): limites del sistema y flujo principal.
- [docs/process-runbooks.md](docs/process-runbooks.md): pasos operativos comunes.
- [docs/data-pipeline.md](docs/data-pipeline.md): pipeline de datos.
- [docs/training.md](docs/training.md): entrenamiento.
- [docs/inference-evaluation.md](docs/inference-evaluation.md): inferencia y evaluacion.

## Guia de instalacion

El proyecto esta estandarizado en Python `3.12`.

Crear un ambiente local nuevo:

```bash
conda create -n congestion-bogota-py312 python=3.12 -y
conda activate congestion-bogota-py312
python -m pip install uv
uv sync --all-groups
```

Si los paquetes de extension de PyTorch Geometric intentan compilarse localmente, instala ruedas precompiladas compatibles con la version local de PyTorch. Para el ambiente CPU validado actualmente:

```bash
python -m pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.9.0+cpu.html
```

Validar la instalacion:

```bash
make test
make lint
```

Si se cuenta con acceso AWS, los datos fuente pueden sincronizarse desde S3:

```bash
uv run -- dotenv run -- aws s3 sync \
  s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/ \
  ./data
```

Los datos grandes y artefactos generados no se versionan en Git. Deben ubicarse en `data/`, `datos/`, `datos_procesados/`, EFS o S3, segun el flujo de trabajo.

## Dependencias

Dependencias principales:

- Python `3.12`.
- PyTorch.
- PyTorch Geometric y PyTorch Geometric Temporal.
- Pandas, NumPy, SciPy, HDF5 y PyArrow.
- GeoPandas, Shapely, OSMnx y NetworkX.
- AWS SDK, AWS CLI y SageMaker para entrenamiento administrado.
- Ray y Dask para flujos distribuidos o de procesamiento intensivo.

Las dependencias completas estan declaradas en [pyproject.toml](pyproject.toml) y bloqueadas en [uv.lock](uv.lock).

## Estructura del repositorio

```text
.
├── src/                 # Paquete principal: preprocessing, dataset, modelos, training e inferencia
├── tests/               # Pruebas unitarias y de comportamiento
├── notebooks/           # Lanzadores SageMaker y notebooks exploratorios
├── scripts/             # Scripts operativos, evaluacion y artefactos
├── docs/                # Documentacion tecnica y operativa
├── ai-context/          # Notas de handoff y estado de experimentos
├── white_paper/         # Fuente LaTeX del white paper del proyecto
├── pyproject.toml       # Dependencias y configuracion de herramientas
├── uv.lock              # Resolucion bloqueada de dependencias
├── Makefile             # Comandos locales comunes
└── Dockerfile.sagemaker # Receta de imagen para SageMaker
```

## Como contribuir

Para contribuir:

1. Revisa la documentacion en [docs/README.md](docs/README.md) y el estado actual en [ai-context/00-start-here.md](ai-context/00-start-here.md).
2. Crea una rama descriptiva para el cambio.
3. Mantiene los cambios acotados al modulo correspondiente.
4. Ejecuta validaciones locales antes de abrir un pull request:

```bash
make test
make lint
```

Las contribuciones deben preservar los contratos principales del sistema:

- un batch de entrenamiento usa un solo grafo local asociado a un seed;
- `edge_index` debe corresponder al conjunto local de nodos;
- el orden de features debe coincidir con el esperado por el modelo;
- los artefactos de entrenamiento, calibracion e inferencia deben documentar su ventana temporal y configuracion.

## Codigo de conducta

Este repositorio no incluye todavia un archivo `CODE_OF_CONDUCT.md`. Se recomienda agregar uno siguiendo las plantillas publicas de GitHub o los lineamientos institucionales aplicables antes de abrir el proyecto a contribuciones externas amplias.

Mientras tanto, se espera que toda interaccion tecnica sea respetuosa, colaborativa y orientada a resolver problemas de manera transparente.

## Autores

Proyecto desarrollado en el marco de trabajo asociado al sistema de prediccion de congestion para Bogota y su documentacion tecnica. Antes de publicar nombres personales o correos, validar la informacion institucional correspondiente.

## Informacion adicional

Documentacion recomendada para nuevos mantenedores:

- [docs/repository-structure.md](docs/repository-structure.md): mapa del codigo.
- [docs/artifacts-and-contracts.md](docs/artifacts-and-contracts.md): contratos de datos, modelos y artefactos.
- [docs/code-reference.md](docs/code-reference.md): inventario de clases y funciones.
- [docs/validation.md](docs/validation.md): senales de calidad y validacion.
- [docs/cleanup-plan.md](docs/cleanup-plan.md): tareas pendientes y riesgos de mantenimiento.

El white paper tecnico se encuentra en [white_paper/traffic_forecasting.tex](white_paper/traffic_forecasting.tex), con referencias en [white_paper/references.bib](white_paper/references.bib).

## Licencia

La licencia del proyecto esta incluida en [LICENSE.md](LICENSE.md). Consulta ese archivo para conocer los permisos, condiciones y restricciones de uso, modificacion y distribucion de la herramienta.

## Limitacion de responsabilidades

El BID no sera responsable, bajo circunstancia alguna, de dano ni indemnizacion, moral o patrimonial; directo o indirecto; accesorio o especial; o por via de consecuencia, previsto o imprevisto, que pudiese surgir:

i. Bajo cualquier teoria de responsabilidad, ya sea por contrato, infraccion de derechos de propiedad intelectual, negligencia o bajo cualquier otra teoria; y/o

ii. A raiz del uso de la Herramienta Digital, incluyendo, pero sin limitacion de potenciales defectos en la Herramienta Digital, o la perdida o inexactitud de los datos de cualquier tipo. Lo anterior incluye los gastos o danos asociados a fallas de comunicacion y/o fallas de funcionamiento de computadoras, vinculados con la utilizacion de la Herramienta Digital.

## Acknowledgments / Reconocimientos

**Copyright © [2025]. Inter-American Development Bank ("IDB"). Authorized Use.**
The procedures and results obtained based on the execution of this software are those programmed by the developers and do not necessarily reflect the views of the IDB, its Board of Executive Directors or the countries it represents.

**Copyright © [2025]. Banco Interamericano de Desarrollo ("BID"). Uso Autorizado.**
Los procedimientos y resultados obtenidos con la ejecucion de este software son los programados por los desarrolladores y no reflejan necesariamente las opiniones del BID, su Directorio Ejecutivo ni los paises que representa.

### Support and Usage Documentation / Documentacion de Soporte y Uso

**Copyright © [2025]. Inter-American Development Bank ("IDB").** The Support and Usage Documentation is licensed under the Creative Commons License CC-BY 4.0 license. The opinions expressed in the Support and Usage Documentation are those of its authors and do not necessarily reflect the opinions of the IDB, its Board of Executive Directors, or the countries it represents.

**Copyright © [2025]. Banco Interamericano de Desarrollo (BID).** La Documentacion de Soporte y Uso esta licenciada bajo la licencia Creative Commons CC-BY 4.0. Las opiniones expresadas en la Documentacion de Soporte y Uso son las de sus autores y no reflejan necesariamente las opiniones del BID, su Directorio Ejecutivo ni los paises que representa.

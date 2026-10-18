# TFG: Síntesis de imágenes médicas OCT a partir de máscaras de segmentación usando Deep Learning

Este repositorio contiene el código desarrollado para el Trabajo de Fin de
Grado. El proyecto trabaja con imágenes de tomografía de coherencia óptica
(OCT) y estudia la reconstrucción o síntesis de una imagen intermedia a partir
de imágenes y máscaras de capas retinianas.

El repositorio se publica principalmente para mostrar la organización del
proyecto, el código utilizado y el flujo de entrenamiento y evaluación. No
todas las partes se pueden ejecutar de forma independiente, ya que algunas
evaluaciones utilizan un segmentador preentrenado proporcionado por la
universidad que no se distribuye en este repositorio.

## Objetivos principales

El código permite:

- Cargar el dataset OCT5K y sus máscaras manuales.
- Preparar los datos para entrenar un modelo de síntesis de imagen.
- Entrenar un modelo condicionado por máscaras de segmentación.
- Interpolar máscaras entre cortes consecutivos.
- Evaluar las imágenes generadas mediante SSIM, MSE y L1.
- Evaluar la calidad de la interpolación de máscaras mediante accuracy,
  Dice e IoU.
- Ejecutar evaluaciones adicionales que utilizan un segmentador externo
  preentrenado.

## Estructura del código

```text
train.py              Entrenamiento del modelo de síntesis.
test.py               Script general de evaluación.
dataset.py            Carga, indexación y preparación del dataset.
utils.py              Utilidades generales y gráficas.
config_manager.py     Lectura de configuraciones y checkpoints.
losses.py             Funciones de pérdida.
mask_utils.py         Operaciones de interpolación de máscaras.
semantic_metrics.py   Métricas semánticas y matrices de confusión.
eval1.py              Evaluación de reconstrucción de imágenes.
eval2.py              Evaluación de interpolación de máscaras.
eval3.py              Evaluación semántica con un segmentador externo.
eval4.py              Evaluación de activaciones/capas con un segmentador externo.
```

## Segmentador preentrenado no incluido

El segmentador de cinco capas utilizado en algunas partes del proyecto fue
proporcionado por la universidad y se mantiene fuera de este repositorio.
Por ese motivo no se incluyen:

```text
segmentador_5_capas/
segmentador_utils.py
segmentador.py
prueba.py
infer.py
*.pth
```

El código de `eval3.py` y `eval4.py` se conserva para mostrar el flujo completo
del proyecto, pero ambos scripts necesitan el segmentador preentrenado externo
para ejecutarse. `test.py` importa las cuatro evaluaciones, por lo que la
evaluación completa también requiere ese segmentador.

En consecuencia:

- `train.py` necesita el dataset y las dependencias de Python.
- `eval1.py` y `eval2.py` no necesitan el segmentador privado.
- `eval3.py` y `eval4.py` no son ejecutables sin el segmentador externo.
- `test.py` no puede ejecutar el flujo completo sin ese segmentador.

El repositorio permite revisar la organización y el código del proyecto, pero
no garantiza la reproducibilidad directa de todas las métricas.

## Dataset

El código espera una estructura equivalente a:

```text
OCT_5k/
├── Images/
│   └── Images_Manual/
│       ├── AMD Part1/
│       ├── AMD Part2/
│       ├── DME/
│       ├── Normal Part1/
│       └── Normal Part2/
├── Masks/
│   └── Masks_Manual/
│       ├── Grading_1/
│       ├── Grading_2/
│       └── Grading_3/
└── Boundaries/
    └── Boundaries_Manual/
```

El dataset original contiene imágenes OCT, máscaras de segmentación y
anotaciones de distintas capas retinianas. También puede contener imágenes y
máscaras automáticas, límites en formato CSV y anotaciones para detección.

El dataset no se incluye en este repositorio. Antes de utilizarlo hay que
comprobar sus condiciones de uso, especialmente porque contiene imágenes
médicas y anotaciones asociadas.

## Configuración de rutas

Los scripts utilizan por defecto un directorio `OCT_5k` situado junto al
código. Si el dataset está en otra ubicación, se puede indicar mediante
`--data-root`. También es necesario indicar la ruta del checkpoint cuando se
evalúa un modelo entrenado.

Por ejemplo:

```bash
python train.py --data-root ruta/al/OCT_5k
python test.py --checkpoint ruta/al/checkpoint.pth --data-root ruta/al/OCT_5k
```

## Dependencias

Las dependencias principales utilizadas por el proyecto son:

```text
torch
torchvision
numpy
Pillow
tqdm
matplotlib
segmentation-models-pytorch
albumentations
opencv-python
scipy
pytorch-ignite
```

Las versiones concretas deben ajustarse al sistema operativo y a la versión
de CUDA disponible. Se recomienda utilizar un entorno virtual y no subirlo al
repositorio.

## Ejemplos de ejecución

Ejemplo conceptual de entrenamiento:

```bash
python train.py --name experimento --loss_type SSIM --lr 0.001 --batch_size 128 --epochs 4000
```

Ejemplo conceptual de evaluación:

```bash
python test.py --checkpoint ruta/al/checkpoint.pth --data-root ruta/al/OCT_5k
```

Los nombres exactos de los argumentos se pueden consultar ejecutando:

```bash
python train.py --help
python test.py --help
```

## Alcance del repositorio

Este repositorio tiene finalidad documental y académica. Se publica para
permitir revisar cómo se organizan las fases de entrenamiento, interpolación y
evaluación. La ausencia del segmentador y de sus pesos implica
que algunas métricas y ejecuciones no son reproducibles directamente a partir
de este repositorio.

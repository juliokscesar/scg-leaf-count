# Contagem e detecção de (pequenas) folhas

- [Relatório]()

O código se resume em duas principais ferramentas que podem ser utilizadas pelo arquivo `detect.py` e `analyze.py`.

O código para os plots estão no notebook [plots.ipynb](https://github.com/juliokscesar/scg-leaf-count/blob/main/plots.ipynb).

## Modelos treinados

Os modelos 0 e 1 treinados localmente podem ser baixados do Google Drive: [modelo 0](https://drive.google.com/file/d/10U50eUNqP-LpBuxqq1WpyzLIWs4GUl9I/view?usp=sharing), [modelo 1](https://drive.google.com/file/d/1OhsZ5W90XB8MN7q7lssfM8Wml5X6aT18/view?usp=sharing).

## Como realizar detecções

Para realizar uma detecção em uma imagem, basta usar `python detect.py <imagem>` que irá printar a quantidade de objetos detectados e plotar a imagem com detecções. Por padrão, o script usa o modelo yolov8n.pt, que é baixado automaticamente.

`python detect.py --help` mostra todas as flags disponíveis para customizar as configurações:

```
usage: detect.py [-h] [--model-type {yolo,roboflow}] [--model-path MODEL_PATH] [-c CONFIDENCE] [-o OVERLAP] [-s] [--slice-w SLICE_W] [--slice-h SLICE_H] [--slice-overlap SLICE_OVERLAP] [--save] [--no-show] img_path

positional arguments:
  img_path              Path to image

options:
  -h, --help            show this help message and exit
  --model-type {yolo,roboflow}
                        Model Type. Default is 'yolo'. Possible options are: yolo, roboflow
  --model-path MODEL_PATH
                        Path to model. Default uses yolov8n.pt
  -c CONFIDENCE, --confidence CONFIDENCE
                        Confidence parameter. Default is 50.0
  -o OVERLAP, --overlap OVERLAP
                        Overlap parameter. Default is 50.0
  -s, --slice-detect    Use slice detection
  --slice-w SLICE_W     Slice width when using slice detection. Default is 640
  --slice-h SLICE_H     Slice height when using slice detection. Default is 640
  --slice-overlap SLICE_OVERLAP
                        Slice overlap ratio when using slice detection. Default is 50.0
  --save                Save image with detections
  --no-show             Don't plot image with detections
```

## Como realizar as contagens

A ferramenta `analyze.py` utiliza a `detect.py` e disponibiliza uma maneira de gerar a contagem de uma ou mais imagens, ou até de um diretório contendo as imagens. Para utilizá-la com, por exemplo, o diretório `imgs/cropped`, basta executar: `python analyze.py imgs/cropped/`.

Todas as configurações utilizadas (como tipo do modelo, local do modelo, parâmetros de detecção, etc.) são retiradas do arquivo `analyze_config.yaml`, que pode ser customizado com suas preferências.

Executando `python analyze.py --help` mostra todas as opções disponíveis na parte de análise.

```
usage: analyze.py [-h] [-c] [--no-show] [--save-detections] images_src [images_src ...]

positional arguments:
  images_src         Source of images. Can be a single or a list of files, or a directory.

options:
  -h, --help         show this help message and exit
  -c, --cached       Use cached data in CSV file. If not specified, avoids having to run detection again and just uses data from before.
  --no-show          Only save data to CSV and don't plot.
  --save-detections  Save image annotated with bounding boxes after detecting.
```


# deteccao-copias-ia

## Base de dados
* Subconjunto do dataset CelebsA

## Processo de detecção de cópias
* Seleção de imaggens
* Extrair faces das imagens
* Gerar embeddings das imagens
* Identificação das imagens duplicadas

## Modelo utilizado
* MTCNN: responsável por extrair as faces das imagens
* VGG-Faces: modelo pré-treinado da biblioteca keras e responsável por gerar embeddings das imagens

## Como instalar

Com o terminal aberto na raiz do projeto, execute os seguintes comandos para criar a env:

```
$ virtualenv --python=python3.6 <env>
$ source <env>/bin/activate
```

Com a env criada e ativada, faça a instalação dos pacotes necessários para executar o script:

```
$ pip3 install -r requirements.txt
$ pip3 install keras==2.2.0
```

## Como executar

Na pasta scripts, executar os seguintes comandos:

1. Seleção das imagens
```
$ python3 ./select_imgs.py -n <número de imagens> -p <caminho das imagens selecionadas>
```

2. Para extrair as faces das imagens
```
$ python3 ./generate_embedding.py -p <caminho das imagens selecionadas>
```

3. Detecção das cópias

Para busca linear:
```
$ python3 ./face_recognition_linear.py
```

Para busca indexada:
```
$ python3 ./face_recognition_index.py
```
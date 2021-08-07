# Quality Enhancement of Highly Degraded Music Using Deep Learning Models

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

[paper](https://www.doi.org/10.1145/3470482.3479635)

Neste trabalho, propomos um método baseado no _Deep-Learning_ para _Audio Inpainting_ acompanhado de um conjunto de dados com condições de fragmentação aleatórias que se aproximam de situações reais de deficiência. O conjunto de dados foi coletado usando faixas de diferentes gêneros musicais para fornecer uma boa variabilidade de sinal. Nosso melhor modelo melhorou a qualidade de todos os gêneros musicais, obtendo uma média de 12,9 dB de PSNR.

## Árvore de arquivos
```
audio_reconstruction/
|----experiments/
|	|----model_save/
|	|----Attention_UNet-t2.ipynb
|	|----Attention_UNet.ipynb
|	|----Efficient_UNet_v2.ipynb
|	|----FPN_RFB_RefineDet.ipynb
|	|----FPN_RFB_RON.ipynb
|	|----FPN_RFB_TDM.ipynb
|	|----FPN.ipynb
|	|----LinkNet.ipynb
|	|----Res_Attention_UNet.ipynb
|	|----Res_UNet_v2.ipynb
|	|----Res_UNet.ipynb
|	|----UNet_RFB_FPN.ipynb
|	|----UNet_RFB_TDM.ipynb
|	|----UNet_v2.ipynb
|	|----UNet.ipynb
|
|----lib/
|	|----models/
|	|	|----ARCNN.py
|	|	|----Attention_FPN_RFB_TDM.py
|	|	|----Attention_UNet.py
|	|	|----Attention_UNet_t2.py
|	|	|----CANDI.py
|	|	|----Card.py
|	|	|----Deeper_SRCNN.py
|	|	|----dnCNN.py
|	|	|----EDSR_ResNet.py
|	|	|----FastARCNN.py
|	|	|----FPN.py
|	|	|----FPN_RFB_RefineDet.py
|	|	|----FPN_RFB_RON.py
|	|	|----FPN_RFB_TDM.py
|	|	|----FPN_RFB_U_Net.py
|	|	|----GRes_SE_ResNet.py
|	|	|----LinkNet.py
|	|	|----ResNet.py
|	|	|----Res_ARCNN.py
|	|	|----Res_Attention_UNet.py
|	|	|----Res_FastARCNN.py
|	|	|----Res_FPN_v2.py
|	|	|----Res_UNet.py
|	|	|----Res_UNet_v2.py
|	|	|----Res_UNet_V2_RFB_TDM.py
|	|	|----SE_ResNet.py
|	|	|----SRResnet.py
|	|	|----UNet.py
|	|	|----UNet_Mandelli.py
|	|	|----UNet_RFB_FPN.py
|	|	|----UNet_RFB_TDM.py
|	|	|----UNet_Sun.py
|	|	|----UNet_v2.py
|	|	|----UNet_V2_RFB_TDM.py
|	|	|----VAE_UNet_v2.py
|	|----evaluate.py
|	|----jpeg.py
|	|----old_utils.py
|	|----test.py
|	|----utils.py
|
|----.gitignore
|----features_extraction_FMA.ipynb
|----LICENSE
|----requirements.txt
|----README.md
```

## Requisitos
Este projeto foi desenvolvido em __Python v3.6.8__, com as bibliotecas listadas no arquivo __requirements.txt__. Para preparar o ambiente basta executar o seguinte comando:

```bash
pip install -r requirements.txt
```

## Código
Os seguintes notebooks, scripts e modelos foram desenvolvidos para o dataset.

1. [`features_extraction_FMA.ipynb`](features_extraction_FMA.ipynb): realiza todo processo de fragmentação dos áudios e salva os respectivos espectrogramas.
2. [`experiments`](https://github.com/TeleMidia/audio_reconstruction/tree/main/src/experiments): modulo com os notebooks de treinamentos com os autoencoders testados para a tarefa.
3. [`lib`](https://github.com/TeleMidia/audio_reconstruction/tree/main/src/lib): modulo com todos os modelos e funcões de avaliação implementadas até então.

## Dataset

#### Descrição do conjunto de dados:

Está base de dados é constituida de espectrogramas extraidos de parte de um conjunto de dados chamado [Free Music Archive (FMA)](https://github.com/mdeff/fma). O mesmo foi produzido para avaliar várias tarefas de aprendiazado com músicas.
Pela facilidade de acesso aos dados e a diversidade de caracteristicas presentes no FMA, utilizou-se o subconjunto 'SMALL' e parte do subconjunto 'MEDIUM' para produção desde.

#### Coleta e processamento de dados:

No total este conjunto é composto por 13.583 áudios de músicas de 16 gêneros com duração de 30 segundos. 
```
    Gêneros:            Quantidade
    Electronic             1637
    Experimental           1624
    Rock                   1608
    Hip-Hop                1585
    Folk                   1518
    Instrumental           1349
    Pop                    1186
    International          1018
    Classical               619
    Historic                510
    Jazz                    384
    Country                 178
    Soul-RnB                154
    Spoken                  118
    Blues                    74
    Easy Listening           21
```

Cada música possui a seguinte configuração de áudio:
```
    Codificador:        PCM S16 LE (s16l)
    Taxa de amostragem: 16000 Hz
    Bits por amostra:   16
    Canais:             mono
```
Dado a relação de complexidade de timbres entre os gêneros foi efetuado a seguinte separação de conjuntos:

```
|    Gêneros    |  Treino   | Validação	|   Teste   |
-----------------------------------------------------
| Electronic    |    800    |	 200	|    637    |
| Experimental  |    800    |	 200	|    624    |
| Rock          |    800    |	 200	|    608    |
| Hip-Hop       |    800    |	 200	|    585    |
| Folk          |    800    |	 200	|    518    |
| Instrumental	|    800    |	 200	|    349    |
| Pop           |    800    |	 200	|    186    |
| International |    800    |	 200	|     18    |
| Classical     |     -     |     -     |    619    |
| Historic      |     -     |	  -     |    510    |
| Jazz          |     -     |	  -     |    384    |
| Country       |     -     |	  -     |    178    |
| Soul-RNB      |     -     |	  -     |    154    |
| Spoken        |     -     |	  -     |    118    |
| Blues         |     -     |	  -     |     74    |
| Easy Listening|     -     |	  -     |     21    |
-----------------------------------------------------
|     TOTAL     |   6400    |   1600	|   5583    |

```

O processo de fragmentação é segmentando em 3 etapas principais, como mostra a seguinte imagem: 
![](https://github.com/TeleMidia/audio_reconstruction/blob/main/figs/partitioning.png?raw=true)

##### Etapa 1
Cada áudio gerou um espectrograma na seguinte dimensão de tempo e frequência respectivamente [7500, 128].

Parâmetros do espectrograma:
```
    rate         = 16000 
    frame_length = 256 
    fft_length   = 255 
    stride       = 64 
    fmin         = 0 
    fmax         = 8000
```
##### Etapa 2
Cada espectrograma foi seguimentado em 58 partes nas dimensões 128x128, gerando os conjuntos nas seguintes dimensões:
```
y_train:
    [371026, 128, 128, 1]
y_val:
    [ 92800, 128, 128, 1]
y_test:
    [323988, 128, 128, 1]
```

Vale observar que as 58 partes __NÃO__ remontam um espectrograma 7500x128, visto que a 59ª parte contém uma ínfima representação temporal do áudio completo, e então foi desconsiderado.

##### Etapa 3
Por fim, cara espectrograma passou por uma fragmentação na dimensão temporal, na qual zerou as frequências em pontos aleatorios. 
Cada espectrograma sofreu uma fragmentação que corresponde entre 10% e 70% das 128 resprensetações temporais produzindo os conjuntos:
```
X_train:
    [371026, 128, 128, 1]
X_val:
    [ 92800, 128, 128, 1]
X_test:
    [323988, 128, 128, 1]
```

Os dados foram salvos em arquivos Numpy (.npy), subtipo float 32 bits, o que possibilita o carregamento de partes do conjunto sob demanda, considerando que apenas carregar todos conjuntos de dados para treinamento ja exigiria entorno de 72Gb de memoria RAM

## Metodologia

O processo de  _Audio Inpainting_ adotado neste trabalho consiste em 2 etapas principais, reconstrução e sintetização. Como descrito na Figura a seguir, o passo de reconstrução consiste em codificar o audio fragmentado em espectrogramas e por conseguinte reconstruir-los atráves de um dos modelos de _autoencoders_ definidos para este projeto. Por fim, a etapa de sintetização é realizada através do algoritmo _Griffin-Lim_, para transformação do espectrograma reconstruído em onda.  

![](https://github.com/TeleMidia/audio_reconstruction/blob/main/figs/method.png?raw=true)


## Projetos Relacionados

- [FMA](https://github.com/mdeff/fma), A Dataset For Music Analysis.

- [Griffin-Lim](https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb),TensorFlow implementation of Griffin-Lim
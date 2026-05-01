# Controlador de Nível de Água por Câmera

Um sistema de detecção de nível de água em tempo real usando visão computacional.

## Características

- ✓ Começa com nível 0 se nenhuma água for detectada
- ✓ Detecta o nível de água pela câmera em tempo real
- ✓ Suavização de leituras com histórico dos últimos 10 frames
- ✓ Interface visual com barra de indicador
- ✓ Status automático (VAZIO, BAIXO, NORMAL, CHEIO)
- ✓ Visualização da máscara de detecção de cor

## Instalação

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Certifique-se de que sua câmera está conectada e funcionando

## Como Usar

Execute o programa:
```bash
python controlador_agua.py
```

## Controles

- **Q** - Sair do programa

## Interpretação dos Resultados

### Status e Cores

- **VAZIO** (Vermelho): 0-10% de água detectada
- **BAIXO** (Laranja): 10-30% de água detectada
- **NORMAL** (Verde): 30-70% de água detectada
- **CHEIO** (Azul): 70-100% de água detectada

### Tela de Saída

A tela mostra:
- **Esquerda**: Vídeo original da câmera com barra de nível
- **Direita**: Máscara binária mostrando o que foi detectado como água

## Como Funciona

1. Captura frames da câmera
2. Converte de BGR para HSV para melhor detecção de cores
3. Identifica pixels com cores típicas de água (tons de azul/verde)
4. Calcula a porcentagem de água na imagem
5. Suaviza os valores usando média dos últimos 10 frames
6. Exibe o nível em tempo real

## Ajustes Possíveis

Para detectar diferentes tipos de água ou cores, edite as faixas HSV em `detectar_agua()`:

```python
# Faixas HSV atuais (azul/verde)
lower_agua = np.array([50, 40, 40])
upper_agua = np.array([130, 255, 255])
```

- **H (Hue)**: 0-180 (cor)
- **S (Saturation)**: 0-255 (pureza da cor)
- **V (Value)**: 0-255 (brilho)

## Requisitos

- Python 3.7+
- Câmera conectada ao computador
- OpenCV 4.8+
- NumPy 1.24+

## Troubleshooting

### Câmera não é detectada
- Verifique se a câmera está conectada
- Teste com outros programas (Windows Camera, etc)
- Mude o índice da câmera em `iniciar_camera()`: `cv2.VideoCapture(1)` ao invés de `0`

### Água não é detectada
- Verifique se a iluminação está adequada
- Ajuste as faixas HSV em `detectar_agua()`
- A água deve estar em tons de azul ou verde

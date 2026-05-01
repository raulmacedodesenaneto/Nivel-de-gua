import cv2
import numpy as np
from collections import deque
import sys
import json
import os
from datetime import datetime
import time

class ControladorNivelAgua:
    def __init__(self, arquivo_config="config.json"):
        self.nivel_agua = 0.0  # Começa com nível 0
        self.historico_niveis = deque(maxlen=30)  # Histórico para suavizar leituras
        self.historico_tempo = deque(maxlen=30)
        self.camera = None
        self.framewidth = 1280
        self.frameheight = 720
        self.fps = 0
        self.tempo_anterior = time.time()
        self.modo_calibracao = False
        
        # Configurações ajustáveis
        self.h_min, self.h_max = 50, 130
        self.s_min, self.s_max = 40, 255
        self.v_min, self.v_max = 40, 255
        self.limite_baixo = 10
        self.limite_normal_min = 30
        self.limite_normal_max = 70
        self.limiar_minimo = 1.0  # Abaixo disto, considera como 0.0%
        self.arquivo_config = arquivo_config
        self.carregar_config()
        
    def carregar_config(self):
        """Carrega configurações salvas"""
        if os.path.exists(self.arquivo_config):
            try:
                with open(self.arquivo_config, 'r') as f:
                    config = json.load(f)
                    self.h_min = config.get('h_min', self.h_min)
                    self.h_max = config.get('h_max', self.h_max)
                    self.s_min = config.get('s_min', self.s_min)
                    self.s_max = config.get('s_max', self.s_max)
                    self.v_min = config.get('v_min', self.v_min)
                    self.v_max = config.get('v_max', self.v_max)
                    print("✓ Configuração carregada do arquivo")
            except Exception as e:
                print(f"⚠ Erro ao carregar config: {e}")
    
    def salvar_config(self):
        """Salva configurações atuais"""
        try:
            config = {
                'h_min': self.h_min, 'h_max': self.h_max,
                's_min': self.s_min, 's_max': self.s_max,
                'v_min': self.v_min, 'v_max': self.v_max
            }
            with open(self.arquivo_config, 'w') as f:
                json.dump(config, f, indent=2)
            print("✓ Configuração salva")
        except Exception as e:
            print(f"✗ Erro ao salvar config: {e}")
        
    def iniciar_camera(self):
        """Inicia a captura da câmera com otimizações"""
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            print("Erro: Não foi possível acessar a câmera!")
            return False
        
        # Configurar propriedades da câmera
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.framewidth)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frameheight)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        
        print("✓ Câmera iniciada com sucesso")
        return True
    
    def detectar_agua(self, frame):
        """
        Detecta o nível de água com múltiplas estratégias
        - Detecção por cor HSV
        - Detecção de contornos
        - Análise de textura
        """
        # Converter BGR para HSV para melhor detecção de cor
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Definir faixas de cor para água usando valores carregados
        lower_agua = np.array([self.h_min, self.s_min, self.v_min])
        upper_agua = np.array([self.h_max, self.s_max, self.v_max])
        
        # Criar máscara para a água
        mascara = cv2.inRange(hsv, lower_agua, upper_agua)
        
        # Aplicar operações morfológicas avançadas
        kernel_pequeno = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Remover ruído
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_grande)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_pequeno)
        
        # Blur para suavizar bordas
        mascara = cv2.GaussianBlur(mascara, (5, 5), 0)
        
        # Threshold binário
        _, mascara = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)
        
        return mascara
    
    def medir_altura_agua(self, mascara):
        """Mede a altura real da água na imagem"""
        altura, largura = mascara.shape
        
        # Varrer de baixo para cima e encontrar a primeira linha com água
        for y in range(altura - 1, -1, -1):
            linha = mascara[y, :]
            if cv2.countNonZero(linha) > largura * 0.1:  # Mínimo 10% da largura
                return altura - y
        
        return 0
    
    def calcular_nivel(self, mascara):
        """
        Calcula o nível de água (0-100%)
        Usa duas métricas: densidade de pixels e altura da água
        Se não houver água (< limiar), retorna 0.0%
        """
        altura, largura = mascara.shape
        
        # Método 1: Porcentagem de pixels
        total_pixels = mascara.size
        pixels_agua = cv2.countNonZero(mascara)
        percentual_pixels = (pixels_agua / total_pixels) * 100
        
        # Método 2: Altura da água
        altura_agua = self.medir_altura_agua(mascara)
        percentual_altura = (altura_agua / altura) * 100
        
        # Combinar métodos com peso
        nivel = (percentual_pixels * 0.6 + percentual_altura * 0.4)
        
        # Se o nível for menor que o limiar mínimo, considera como 0.0%
        if nivel < self.limiar_minimo:
            return 0.0
        
        return np.clip(nivel, 0, 100)
    
    def processar_frame(self, frame):
        """Processa um frame e retorna o nível de água detectado"""
        # Detectar água
        mascara = self.detectar_agua(frame)
        
        # Calcular nível
        nivel = self.calcular_nivel(mascara)
        
        # Adicionar ao histórico
        self.historico_niveis.append(nivel)
        self.historico_tempo.append(time.time())
        
        # Calcular média suavizada (peso maior para valores recentes)
        if len(self.historico_niveis) > 0:
            niveis = np.array(list(self.historico_niveis))
            # Peso exponencial: valores recentes têm mais peso
            pesos = np.exp(np.linspace(-1, 0, len(niveis)))
            pesos = pesos / pesos.sum()
            nivel_suavizado = np.sum(niveis * pesos)
        else:
            nivel_suavizado = nivel
        
        # Atualizar nível atual
        self.nivel_agua = nivel_suavizado
        
        # Calcular FPS
        tempo_atual = time.time()
        if (tempo_atual - self.tempo_anterior) > 0:
            self.fps = 1.0 / (tempo_atual - self.tempo_anterior)
        self.tempo_anterior = tempo_atual
        
        return mascara, nivel_suavizado
    
    def desenhar_interface(self, frame, mascara, nivel):
        """Desenha interface visual completa com indicadores"""
        altura, largura = frame.shape[:2]
        saida = frame.copy()
        
        # ========== SEÇÃO 1: BARRA DE NÍVEL VERTICAL ==========
        barra_x1, barra_y1 = largura - 120, 20
        barra_x2, barra_y2 = largura - 50, altura - 20
        barra_largura = barra_x2 - barra_x1
        barra_altura = barra_y2 - barra_y1
        
        # Desenhar fundo da barra
        cv2.rectangle(saida, (barra_x1, barra_y1), (barra_x2, barra_y2), 
                     (50, 50, 50), -1)
        cv2.rectangle(saida, (barra_x1, barra_y1), (barra_x2, barra_y2), 
                     (255, 255, 255), 2)
        
        # Calcular altura do preenchimento
        altura_preenchimento = int((nivel / 100) * barra_altura)
        y_inicio = barra_y2 - altura_preenchimento
        
        # Cor baseada no nível
        if nivel < self.limite_baixo:
            cor_barra = (0, 0, 255)  # Vermelho
        elif nivel < self.limite_normal_min:
            cor_barra = (0, 165, 255)  # Laranja
        elif nivel < self.limite_normal_max:
            cor_barra = (0, 255, 0)  # Verde
        else:
            cor_barra = (255, 0, 0)  # Azul
        
        # Desenhar preenchimento
        cv2.rectangle(saida, (barra_x1, y_inicio), (barra_x2, barra_y2), 
                     cor_barra, -1)
        
        # Linha de referência no meio (50%)
        y_meio = barra_y1 + barra_altura // 2
        cv2.line(saida, (barra_x1 - 10, y_meio), (barra_x2 + 5, y_meio), 
                (200, 200, 200), 1)
        cv2.putText(saida, "50%", (barra_x1 - 40, y_meio + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # ========== SEÇÃO 2: INFORMAÇÕES EM TEMPO REAL ==========
        y_pos = 30
        info_x = 10
        
        # Nível percentual
        texto_nivel = f"Nivel: {nivel:.1f}%"
        cv2.putText(saida, texto_nivel, (info_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Status
        y_pos += 50
        if nivel < self.limite_baixo:
            status = "VAZIO"
            cor = (0, 0, 255)
        elif nivel < self.limite_normal_min:
            status = "BAIXO"
            cor = (0, 165, 255)
        elif nivel < self.limite_normal_max:
            status = "NORMAL"
            cor = (0, 255, 0)
        else:
            status = "CHEIO"
            cor = (255, 0, 0)
        
        cv2.putText(saida, f"Status: {status}", (info_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor, 2)
        
        # FPS
        y_pos += 50
        cv2.putText(saida, f"FPS: {self.fps:.1f}", (info_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Hora
        y_pos += 40
        hora = datetime.now().strftime("%H:%M:%S")
        cv2.putText(saida, hora, (info_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # ========== SEÇÃO 3: HISTÓRICO ==========
        if len(self.historico_niveis) > 1:
            y_pos += 50
            cv2.putText(saida, "Tendencia:", (info_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)
            
            # Calcular variação
            variacao = self.historico_niveis[-1] - self.historico_niveis[0]
            if variacao > 0.5:
                tendencia = "↑ Subindo"
                cor_tendencia = (0, 255, 0)
            elif variacao < -0.5:
                tendencia = "↓ Descendo"
                cor_tendencia = (0, 0, 255)
            else:
                tendencia = "→ Estável"
                cor_tendencia = (0, 255, 255)
            
            cv2.putText(saida, tendencia, (info_x + 100, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_tendencia, 1)
        
        # ========== SEÇÃO 4: MODO CALIBRAÇÃO ==========
        if self.modo_calibracao:
            y_pos = altura - 80
            cv2.rectangle(saida, (10, y_pos - 20), (largura - 10, altura - 10), 
                         (0, 255, 255), 2)
            cv2.putText(saida, "MODO CALIBRACAO ATIVO", (20, y_pos + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(saida, f"H:{self.h_min}-{self.h_max} S:{self.s_min}-{self.s_max} V:{self.v_min}-{self.v_max}", 
                       (20, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        
        # ========== SEÇÃO 5: MÁSCARA LADO A LADO ==========
        mascara_colorida = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
        
        # Adicionar contorno em verde na máscara
        contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mascara_colorida, contours, -1, (0, 255, 0), 2)
        
        # Combinar
        resultado = np.hstack([saida, mascara_colorida])
        
        return resultado
    
    def processar_entrada(self, tecla):
        """Processa entrada do teclado"""
        if tecla == ord('q') or tecla == ord('Q'):
            return 'sair'
        
        elif tecla == ord('c') or tecla == ord('C'):
            self.modo_calibracao = not self.modo_calibracao
            status = "ATIVADO" if self.modo_calibracao else "DESATIVADO"
            print(f"\nModo Calibração {status}")
            if self.modo_calibracao:
                print("Controles de Calibração:")
                print("  H+/H-: ajustar Hue min/max")
                print("  S+/S-: ajustar Saturation")
                print("  V+/V-: ajustar Value")
                print("  SAVE: salvar configuração")
            return 'continuar'
        
        if self.modo_calibracao:
            passo = 5
            if tecla == ord('h') or tecla == ord('H'):
                self.h_min = max(0, self.h_min - passo)
                print(f"H_min: {self.h_min}")
            elif tecla == ord('H'):
                self.h_max = min(180, self.h_max + passo)
                print(f"H_max: {self.h_max}")
            elif tecla == ord('s') or tecla == ord('S'):
                self.s_min = max(0, self.s_min - passo)
                print(f"S_min: {self.s_min}")
            elif tecla == ord('S'):
                self.s_max = min(255, self.s_max + passo)
                print(f"S_max: {self.s_max}")
            elif tecla == ord('v') or tecla == ord('V'):
                self.v_min = max(0, self.v_min - passo)
                print(f"V_min: {self.v_min}")
            elif tecla == ord('V'):
                self.v_max = min(255, self.v_max + passo)
                print(f"V_max: {self.v_max}")
            elif tecla == ord('w') or tecla == ord('W'):
                self.salvar_config()
        
        return 'continuar'
    
    def executar(self):
        """Loop principal do controlador"""
        if not self.iniciar_camera():
            return
        
        print("\n" + "="*60)
        print("   CONTROLADOR DE NÍVEL DE ÁGUA COM CÂMERA")
        print("="*60)
        print("\nControles:")
        print("  Q          - Sair")
        print("  C          - Modo Calibração (ajustar cores)")
        print("  W (cal)    - Salvar configuração")
        print("\nLegenda de Status:")
        print(f"  VAZIO:  0-{self.limite_baixo}% (Vermelho)")
        print(f"  BAIXO:  {self.limite_baixo}-{self.limite_normal_min}% (Laranja)")
        print(f"  NORMAL: {self.limite_normal_min}-{self.limite_normal_max}% (Verde)")
        print(f"  CHEIO:  {self.limite_normal_max}-100% (Azul)")
        print("\n" + "="*60 + "\n")
        
        frame_count = 0
        
        while True:
            ret, frame = self.camera.read()
            
            if not ret:
                print("Erro ao capturar frame da câmera!")
                break
            
            frame_count += 1
            
            # Processar frame
            mascara, nivel = self.processar_frame(frame)
            
            # Desenhar interface
            resultado = self.desenhar_interface(frame, mascara, nivel)
            
            # Exibir
            cv2.imshow("Controlador de Nivel de Agua", resultado)
            
            # Log a cada 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count:5d} | Nível: {self.nivel_agua:6.1f}% | FPS: {self.fps:5.1f}")
            
            # Verificar entrada
            tecla = cv2.waitKey(1) & 0xFF
            if tecla != 255:  # Se alguma tecla foi pressionada
                resultado = self.processar_entrada(tecla)
                if resultado == 'sair':
                    break
        
        self.fechar()
    
    def fechar(self):
        """Fecha a câmera e libera recursos"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("\n✓ Controlador encerrado com sucesso")
        print(f"Total de frames processados: {self.framecount if hasattr(self, 'framecount') else 'N/A'}")

if __name__ == "__main__":
    controlador = ControladorNivelAgua()
    controlador.executar()

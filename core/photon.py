import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

class Photon:
  def __init__(self, input):
    self._input= self._source(input)
    self._stream= None

  def _source(self, path):
    #tratar diferentes inputs possíveis (imagem, video, camera)
    return

  def to_hsv(self, previous):
    #recebe uma imagem e converte o padrão "previous" para hsv
    return

  def color_segment(self, threshold):
    #recebe uma imagem hsv
    #cria uma máscara binária
    #segmenta todos os pixels de uma imagem que esteja dentro de um intervalo de cor
    return

  def morph(self, operation):
    #implementar operações morfológicas
    return

  def opening(self, kernel):
    #implementar o algoritmo de abertura.

    return

  def closing(self, kernel):
    #implementar o algoritmo de fechamento.

    return

  def bilateral_filter(self, args):
    #aplicação do filtro bilateral
    return


  def clahe(self):
    #versão evoluída do algoritmo do histograma
    return

  def lightbreak(self):
    #aplica limiarização adaptativa
    #separa regiões claras e escuras
    return

  def border_canny(self):
    #algoritmo canny de detecção de borda
    return


  def surfaceMap():
    #recebe uma máscara binária limpa e retorna uma lista de coordenadas dos planos detectados.
    return

  def robot_eye():
    #engloba as funções anteriores na intenção de criar uma função definitiva
    return

  def orb_reveal(self):
    message = ""
    #implementar descritor de imagem
    #descreve a iluminação do local e ângulo de origem da luz.
    #retorna algum aviso sobre obstáculos ou riscos.

    return message


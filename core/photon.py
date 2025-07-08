import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import requests
import random

class Photon:
  def __init__(self, input):
    if isinstance(input, np.ndarray):
        self._stream = input
        self._input = input
    else:
        self._input = self._source(input)
        self._stream = None

  
  @staticmethod
  def from_pokemon(name):
    url = f"https://pokeapi.co/api/v2/pokemon/{name.lower()}"
    resp = requests.get(url)

    if resp.status_code == 200:
      data = resp.json()
      img_url = data["sprites"]["other"]["official-artwork"]["front_default"]
      img_resp = requests.get(img_url)
      img_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
      image = cv.imdecode(img_array, cv.IMREAD_COLOR)
      photon = Photon(image)
      return photon
    else:
      raise Exception("Pokémon não encontrado.")

  @staticmethod
  def from_random_pokemon():
    max_id = 898  # Total de Pokémon na PokéAPI (até a 8ª geração)
    random_id = random.randint(1, max_id)

    url = f"https://pokeapi.co/api/v2/pokemon/{random_id}"
    resp = requests.get(url)

    if resp.status_code == 200:
      data = resp.json()
      img_url = data["sprites"]["other"]["official-artwork"]["front_default"]

      if img_url is None:
        raise Exception("Pokémon não possui imagem oficial.")

      img_resp = requests.get(img_url)
      img_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
      image = cv.imdecode(img_array, cv.IMREAD_COLOR)

      if image is None:
        raise Exception("Falha ao carregar imagem do Pokémon.")

      photon = Photon(image)
      photon._stream = image
      return photon
    else:
      raise Exception("Erro ao buscar Pokémon aleatório.")
    

  @staticmethod
  def from_magic_card(name):
    headers = {
      "User-Agent": "PhotonApp/1.0",
      "Accept": "*/*"
    }
    url = f"https://api.scryfall.com/cards/named?fuzzy={name}"
    resp = requests.get(url, headers=headers)

    if resp.status_code == 200:
      img_url = resp.json()["image_uris"]["normal"]
      img_resp = requests.get(img_url)
      img_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
      image = cv.imdecode(img_array, cv.IMREAD_COLOR)
      photon = Photon(image)
      photon._stream = image
      return photon
    else:
      raise Exception("Carta não encontrada.")

  @staticmethod
  def from_random_magic_card():
    headers = {
      "User-Agent": "PhotonApp/1.0",
      "Accept": "*/*"
    }
    url = "https://api.scryfall.com/cards/random"
    resp = requests.get(url, headers=headers)

    if resp.status_code == 200:
      img_url = resp.json()["image_uris"]["normal"]
      img_resp = requests.get(img_url)
      img_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
      image = cv.imdecode(img_array, cv.IMREAD_COLOR)
      photon = Photon(image)
      photon._stream = image
      return photon
    else:
      raise Exception("Erro ao buscar carta aleatória.")
    
  def show(self, window_name="Imagem"):
    if self._stream is not None:
      cv.imshow(window_name, self._stream)
      cv.waitKey(0)
      cv.destroyAllWindows()
    else:
      print("Nenhuma imagem carregada.")

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


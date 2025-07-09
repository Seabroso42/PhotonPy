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
  def from_magic_card(name, image_type="normal"):
    """
    Busca uma carta do Magic pelo nome. 
    `image_type` pode ser: "normal", "large", "png", "art_crop", "border_crop"
    """
    headers = {
        "User-Agent": "PhotonApp/1.0",
        "Accept": "*/*"
    }
    url = f"https://api.scryfall.com/cards/named?fuzzy={name}"
    resp = requests.get(url, headers=headers)

    if resp.status_code == 200:
        data = resp.json()

        if "image_uris" not in data or image_type not in data["image_uris"]:
            raise Exception(f"Tipo de imagem '{image_type}' não disponível para esta carta.")

        img_url = data["image_uris"][image_type]
        img_resp = requests.get(img_url)
        img_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
        image = cv.imdecode(img_array, cv.IMREAD_COLOR)
        photon = Photon(image)
        photon._stream = image
        return photon
    else:
        raise Exception("Carta não encontrada.")
  @staticmethod
  def from_random_magic_card(image_type="normal"):
    headers = {
      "User-Agent": "PhotonApp/1.0",
      "Accept": "*/*"
    }
    url = "https://api.scryfall.com/cards/random"
    resp = requests.get(url, headers=headers)

    if resp.status_code == 200:
      img_url = resp.json()["image_uris"][image_type]
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

  def show_side_by_side(self, other_image, title1="Original", title2="Processada"):
    """
    Exibe duas imagens lado a lado usando matplotlib.
    `other_image` deve ser um np.ndarray ou Photon com ._stream.
    """
    if self._stream is None:
        print("Imagem principal não carregada.")
        return

    if isinstance(other_image, Photon):
        img2 = other_image._stream
    elif isinstance(other_image, np.ndarray):
        img2 = other_image
    else:
        print("Formato de imagem inválido para comparação.")
        return

    img1_rgb = cv.cvtColor(self._stream, cv.COLOR_BGR2RGB)
    img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_rgb)
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2_rgb)
    plt.title(title2)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

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

  def bilateral_filter(self, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Aplica o filtro bilateral para redução de ruído preservando bordas.
    d: diâmetro do pixel vizinho considerado
    sigmaColor: influência das diferenças de intensidade
    sigmaSpace: influência da distância entre pixels
    """
    if self._stream is not None:
        self._stream = cv.bilateralFilter(self._stream, d, sigmaColor, sigmaSpace)
    return self
  
  def apply_median_filter(self, kernel_size=5):
    """
    Aplica o filtro de mediana para redução de ruído.
    Indicado para remover ruído do tipo sal e pimenta.
    """
    if self._stream is not None:
        self._stream = cv.medianBlur(self._stream, kernel_size)
    return self

  def clahe(self, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Aplica CLAHE (equalização adaptativa de histograma) para melhorar o contraste.
    Funciona em imagens coloridas convertendo para o espaço YCrCb e aplicando no canal Y (luminância).
    """
    if self._stream is not None:
        ycrcb = cv.cvtColor(self._stream, cv.COLOR_BGR2YCrCb)

        y, cr, cb = cv.split(ycrcb)

        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        y_eq = clahe.apply(y)

        ycrcb_eq = cv.merge((y_eq, cr, cb))

        self._stream = cv.cvtColor(ycrcb_eq, cv.COLOR_YCrCb2BGR)

    return self  

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


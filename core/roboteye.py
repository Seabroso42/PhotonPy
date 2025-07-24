import numpy as np
import cv2 as cv
import requests
import random
import os
from typing import *
import matplotlib.pyplot as plt

class RobotEye:
  def __init__(self,outpath:Optional[str], connection:Union[None,int, str, cv.Mat] = None):
      self._connection = cv.VideoCapture(connection)
      self._outpath = outpath
      self._source_type = self._determine_source_type(connection)
      self._capture = None
      self._validate_output_path(outpath)
      self._capture_device = None
      self._current_frame = None
      self._setup_source(source)

def _validate_output_path(self, path: str):
    """Cria o diretório de saída se não existir"""
    os.makedirs(path, exist_ok=True)

def _determine_source_type(self, source) -> str:
        """Identifica o tipo de fonte de imagem"""
        if source is None:
            return 'camera'
        elif isinstance(source, int):
            return 'camera'
        elif isinstance(source, str):
            return 'file'
        elif isinstance(source, cv.Mat):
            return 'direct'
        else:
            raise ValueError("Tipo de fonte não suportado")

def _setup_source(self, source):
        """Configura a fonte de imagem conforme o tipo"""
        if self._source_type == 'camera':
            self._init_camera(source if isinstance(source, int) else 0)
        elif self._source_type == 'file':
            self._load_image_file(source)
        elif self._source_type == 'direct':
            self._current_frame = source

def _init_camera(self, camera_index: int):
        """Inicializa a captura da câmera"""
        self._capture_device = cv2.VideoCapture(camera_index)
        if not self._capture_device.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera {camera_index}")
        self._update_frame()

def _load_image_file(self, file_path: str):
        """Carrega imagem do arquivo"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        self._current_frame = cv2.imread(file_path)
        if self._current_frame is None:
            raise ValueError(f"Falha ao ler imagem: {file_path}")

def _update_frame(self):
        """Atualiza o frame atual (para câmeras)"""
        if self._source_type == 'camera':
            ret, frame = self._capture_device.read()
            if ret:
                self._current_frame = frame
            else:
                raise RuntimeError("Erro ao capturar frame da câmera")

#input treatment
@staticmethod
def pokefetch(name: str = None) -> np.ndarray:
    try:
        url = (f"https://pokeapi.co/api/v2/pokemon/{name.lower()}"
              if name else
              f"https://pokeapi.co/api/v2/pokemon/{random.randint(1, 1025)}")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        img_url = (data.get("sprites", {}).get("other", {}).get("official-artwork", {}).get("front_default")
                  or data.get("sprites", {}).get("front_default"))
        if not img_url:
            raise ValueError("No image available")

        img_resp = requests.get(img_url, timeout=10)
        img_resp.raise_for_status()
        img_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise ValueError(f"Failed to fetch Pokémon: {str(e)}")


@staticmethod
def magic_gather(name: str = None, image_type: str = "art_crop") -> np.ndarray:
    try:
        url = (f"https://api.scryfall.com/cards/named?fuzzy={name}"
              if name else
              "https://api.scryfall.com/cards/random")

        resp = requests.get(url, headers={"User-Agent": "PhotonApp/1.0"}, timeout=10)
        resp.raise_for_status()
        card = resp.json()

        img_url = (card.get("image_uris", {}).get(image_type)
                  or card.get("card_faces", [{}])[0].get("image_uris", {}).get(image_type))
        if not img_url:
            raise ValueError("Image type not available")

        img_resp = requests.get(img_url, timeout=10)
        img_resp.raise_for_status()
        img_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise ValueError(f"Failed to fetch Magic card: {str(e)}")

@staticmethod
def readImage(imageName:str):
  imgPath = os.path.join(rootpath, f'computervision/images/{imageName}')
  image = cv.imread(imgPath)
  return image

def readROI(image, initpoint: tuple[int, int], endpoint: tuple [int,int]):
  x1, y1 = initpoint
  x2, y2 = endpoint
  imageRGB= cv.cvtColor(image, cv.COLOR_BGR2RGB)
  height, width = imageRGB.shape[:2]
  x1, y1 = max(0, x1), max(0, y1)
  x2, y2 = min(width, x2), min(height, y2)

  region = imageRGB[y1:y2, x1:x2]

  if region.size == 0:
    print('a área selecionada está vazia ou fora da imagem')
    return

  return region

def returnCapture(self, pattern):
   #cria um objeto photon com um padrão de cor especificado
   match pattern:
      case 'bgr':
         self._capture = cv.cvtColor()
   return self._capture


#def readToYOLO() -> box // normalize // convert

@staticmethod
def readNshow(imageName):
  imgPath = os.path.join(rootpath, f'computervision/images/{imageName}')
  image = cv.imread(imgPath)
  imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)

  plt.figure()
  plt.imshow(imageRGB)
  plt.show()

def videoWebcam():
  cap = cv.VideoCapture(0)

  if not cap.isOpened():
    exit()
  while True:
    ret, frame = cap.read()
    if ret:
      cv.imshow('webcam', frame)
    if cv.waitKey(1) == ord('q'):
      break

  cap.release()
  cv.destroyAllWindows()

def WebcamToFile():
  cap = cv.VideoCapture(0)
  fourcc = cv.VideoWriter_fourcc(*'XVID')
  outPath = os.path.join(outpath, 'webcam.avi')
  out = cv.VideoWriter(outPath, fourcc, 20.0, (640, 480))

  while cap.isOpened():
    ret, frame = cap.read()
    if ret:
      out.write(frame)
      cv.imshow("webcam-to-file", frame)
    if cv.waitKey(1) == ord('q'):
      break

def VideoFromFile(vidName):
  vidPath= os.path.join(rootpath, f'computervision/videos/{vidName}')
  cap = cv.VideoCapture(vidPath)
  while cap.isOpened():
    ret, frame = cap.read()
    cv.imshow(f'{vidName}', frame)
    #depende de como o programa interpreta o video (no caso 1000milisec)
    delay = int(1000/60)
    if cv.waitKey(delay) == ord('q'):
      break

  cap.release()
  cv.destroyAllWindows()
def saveImage(image, savingName):
  outname = os.path.join(outpath, savingName)
  cv.imwrite(outname, image)


import numpy as np
import cv2 as cv
import random
from dataclasses import dataclass
import matplotlib
import matplotlib.pyplot as plt
from typing import *

@dataclass
class ColorPattern:
    name:str
    channels: int
    opencv_code: Optional[int] = None

class Photon:
    def __init__(self, input:Union[np.ndarray,cv.Mat], colorPattern:Union[ColorPattern,str]='bgr'):
        self._purity = input.copy()
        self._processed = None
        self._colorPattern = colorPattern
        self._model = None #classe que encapsula os modelos ML
        self._histogram = None #classe que facilita o acesso aos dados de histograma
        self._validate_image()
      #  self._color_pattern = self._init_color_pattern(color_pattern)
        self._model_runner = None
        self._processing_stack = []

    @property
    def original(self) -> np.ndarray:
        """Acessa a imagem original"""
        return self._original_image.copy()

    @property
    def processed(self) -> np.ndarray:
        """Acessa a imagem processada"""
        return self._processed_image.copy()

    @property
    def histogram(self):
        """Acessa o histograma (calculado sob demanda)"""
        if self._histogram is None:
            self._calculate_histogram()
        return self._histogram

    def copy(self):
        new_photon = Photon(np.copy(self._input))
        new_photon._stream = np.copy(self._stream)
        return new_photon

    def _validate_image(self, image: np.ndarray):
        """Verifica se a imagem é válida"""
        if not isinstance(image, np.ndarray):
            raise TypeError("Imagem deve ser um array numpy")
        if image.size == 0:
            raise ValueError("Imagem vazia")

    def _init_color_pattern(self, pattern) -> ColorPattern:
        """Inicializa o padrão de cores"""
        patterns = {
            'bgr': ColorPattern('bgr', 3, None),
            'rgb': ColorPattern('rgb', 3, cv2.COLOR_BGR2RGB),
            'hsv': ColorPattern('hsv', 3, cv2.COLOR_BGR2HSV),
            'grayscale': ColorPattern('grayscale', 1, cv2.COLOR_BGR2GRAY)
        }

        if isinstance(pattern, ColorPattern):
            return pattern
        elif pattern in patterns:
            return patterns[pattern]
        else:
            raise ValueError(f"Padrão de cores não suportado: {pattern}")


    def show_side_by_side(self, other, title1="Original", title2="Processed", save_path="comparison.png"):
        if self._stream is None or other._stream is None:
            return

        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(cv.cvtColor(self._stream, cv.COLOR_BGR2RGB))
        ax1.set_title(title1)
        ax1.axis('off')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(cv.cvtColor(other._stream, cv.COLOR_BGR2RGB))
        ax2.set_title(title2)
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Resultado salvo em: {save_path}")
        plt.close(fig)

    # Métodos de pré-processamento
    def clahe(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        if self._stream is not None:
            gray = cv.cvtColor(self._stream, cv.COLOR_BGR2GRAY) if len(self._stream.shape) == 3 else self._stream
            clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            self._stream = clahe.apply(gray)
        return self

    def bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        if self._stream is not None:
            self._stream = cv.bilateralFilter(self._stream, d, sigma_color, sigma_space)
        return self

    def apply_median_filter(self, kernel_size=5):
        if self._stream is not None:
            self._stream = cv.medianBlur(self._stream, kernel_size)
        return self

    def morph(self, operation, ksize=3, kshape='rect'):
        if self._stream is not None:
            shape_map = {'rect': cv.MORPH_RECT, 'ellipse': cv.MORPH_ELLIPSE, 'cross': cv.MORPH_CROSS}
            kernel = cv.getStructuringElement(shape_map[kshape], (ksize, ksize))

            if operation == 'open':
                self._stream = cv.morphologyEx(self._stream, cv.MORPH_OPEN, kernel)
            elif operation == 'close':
                self._stream = cv.morphologyEx(self._stream, cv.MORPH_CLOSE, kernel)
        return self

    def binarize(self, method='histogram', threshold=None):
        if self._stream is None:
            return self

        gray = cv.cvtColor(self._stream, cv.COLOR_BGR2GRAY) if len(self._stream.shape) == 3 else self._stream

        if method == 'histogram':
            threshold = threshold if threshold is not None else np.mean(gray)
            _, self._stream = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)

        elif method == 'otsu':
            _, self._stream = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        elif method == 'adaptive':
            self._stream = cv.adaptiveThreshold(
                gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY, 11, threshold if threshold is not None else 2
            )

        return self
import numpy as np
import cv2 as cv
import requests
import random
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo garantido
import matplotlib.pyplot as plt

class Photon:
    def __init__(self, input):
        if isinstance(input, np.ndarray):
            self._stream = input.copy()
            self._input = input.copy()
        else:
            self._input = self._source(input)
            self._stream = self._input.copy() if self._input is not None else None

    @staticmethod
    def pokefetch(name: str = None) -> 'Photon':
        try:
            url = f"https://pokeapi.co/api/v2/pokemon/{name.lower()}" if name else f"https://pokeapi.co/api/v2/pokemon/{random.randint(1, 1025)}"
            resp = requests.get(url)
            resp.raise_for_status()
            img_url = resp.json().get("sprites", {}).get("other", {}).get("official-artwork", {}).get("front_default") or resp.json().get("sprites", {}).get("front_default")
            if not img_url:
                raise ValueError("No image available")

            img_resp = requests.get(img_url)
            img_resp.raise_for_status()
            img_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
            image = cv.imdecode(img_array, cv.IMREAD_COLOR)
            return Photon(image)
        except Exception as e:
            raise ValueError(f"Failed to fetch Pokémon: {str(e)}")

    @staticmethod
    def magic_gather(name: str = None, image_type: str = "art_crop") -> 'Photon':
        try:
            url = f"https://api.scryfall.com/cards/named?fuzzy={name}" if name else "https://api.scryfall.com/cards/random"
            resp = requests.get(url, headers={"User-Agent": "PhotonApp/1.0"})
            resp.raise_for_status()
            card = resp.json()
            img_url = card.get("image_uris", {}).get(image_type) or card.get("card_faces", [{}])[0].get("image_uris", {}).get(image_type)
            if not img_url:
                raise ValueError(f"Image type '{image_type}' not available")

            img_resp = requests.get(img_url)
            img_resp.raise_for_status()
            img_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
            image = cv.imdecode(img_array, cv.IMREAD_COLOR)
            return Photon(image)
        except Exception as e:
            raise ValueError(f"Failed to fetch Magic card: {str(e)}")

    def _source(self, path):
        return cv.imread(path)

    def copy(self):
        new_photon = Photon(np.copy(self._input))
        new_photon._stream = np.copy(self._stream)
        return new_photon

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
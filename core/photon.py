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
    def pokefetch(name: str = None) -> 'Photon':
        """
        Fetch Pokémon image from PokeAPI.

        Args:
            name (str, optional): Pokémon name. If None, returns random Pokémon.

        Returns:
            Photon: Photon object with Pokémon image

        Raises:
            Exception: If Pokémon not found or image unavailable
        """
        if name:
            url = f"https://pokeapi.co/api/v2/pokemon/{name.lower()}"
            error_msg = f"Pokémon '{name}' não encontrado."
        else:
            max_id = 1025
            random_id = random.randint(1, max_id)
            url = f"https://pokeapi.co/api/v2/pokemon/{random_id}"
            error_msg = "Erro ao buscar Pokémon aleatório."

        resp = requests.get(url)
        if resp.status_code != 200:
            raise Exception(f"{error_msg} (Status: {resp.status_code})")

        data = resp.json()
        img_url = data.get("sprites", {}).get("other", {}).get("official-artwork", {}).get("front_default")

        if not img_url:
            img_url = data.get("sprites", {}).get("front_default")
            if not img_url:
                raise Exception("Pokémon encontrado, mas não possui imagem disponível.")

        img_resp = requests.get(img_url)
        if img_resp.status_code != 200:
            raise Exception("Falha ao fazer o download da imagem do Pokémon.")

        img_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
        image = cv.imdecode(img_array, cv.IMREAD_COLOR)

        if image is None:
            raise Exception("Falha ao decodificar a imagem do Pokémon.")

        return Photon(image)

    @staticmethod
    def magic_gather(name: str = None, image_type: str = "art_crop"):
        """
        Fetch Magic: The Gathering cards from Scryfall API.

        Args:
            name (str, optional): Card name or set name. If None, returns random card.
            image_type (str): Image type. Options: "normal", "large", "png",
                            "art_crop", "border_crop". Default "art_crop".

        Returns:
            Photon: Photon object with card image

        Raises:
            ValueError: For invalid parameters or when card/image not found
        """
        valid_image_types = ["normal", "large", "png", "art_crop", "border_crop"]
        if image_type not in valid_image_types:
            raise ValueError(f"image_type must be one of {valid_image_types}")

        headers = {
            "User-Agent": "PhotonApp/1.0",
            "Accept": "*/*"
        }

        try:
            if name:
                search_url = f"https://api.scryfall.com/cards/search?q={name}"
                search_resp = requests.get(search_url, headers=headers)

                if search_resp.status_code == 200:
                    cards = search_resp.json().get('data', [])
                    if cards:
                        card = random.choice(cards)
                    else:
                        raise ValueError("No cards found")
                else:
                    fuzzy_url = f"https://api.scryfall.com/cards/named?fuzzy={name}"
                    fuzzy_resp = requests.get(fuzzy_url, headers=headers)
                    if fuzzy_resp.status_code == 200:
                        card = fuzzy_resp.json()
                    else:
                        random_resp = requests.get("https://api.scryfall.com/cards/random", headers=headers)
                        if random_resp.status_code == 200:
                            card = random_resp.json()
                        else:
                            raise ValueError("No cards found and failed to get random card")
            else:
                random_resp = requests.get("https://api.scryfall.com/cards/random", headers=headers)
                if random_resp.status_code == 200:
                    card = random_resp.json()
                else:
                    raise ValueError("Failed to get random card")

            if "image_uris" in card:
                image_url = card["image_uris"].get(image_type)
            elif "card_faces" in card:
                image_url = card["card_faces"][0]["image_uris"].get(image_type)
            else:
                raise ValueError(f"Image type '{image_type}' not available for this card")

            if not image_url:
                raise ValueError(f"Image type '{image_type}' not available for this card")

            img_resp = requests.get(image_url, headers=headers)
            img_resp.raise_for_status()

            img_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
            cv_image = cv.imdecode(img_array, cv.IMREAD_COLOR)

            photon = Photon(cv_image)
            photon._stream = cv_image
            return photon

        except requests.exceptions.RequestException as e:
            raise ValueError(f"API request failed: {str(e)}")

    def show(self, window_name="Imagem"):
        """Display image using OpenCV."""
        if self._stream is not None:
            cv.imshow(window_name, self._stream)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print("Nenhuma imagem carregada.")

    def show_side_by_side(self, other_image, title1="Original", title2="Processada"):
        """
        Display two images side by side using matplotlib.

        Args:
            other_image: Photon object or numpy array
            title1: Title for first image
            title2: Title for second image
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
        """Handle different input sources (image, video, camera)."""
        return cv.imread(path)

    def to_hsv(self, previous):
        """Convert image to HSV color space."""
        if self._input is not None:
            if previous.lower() == "bgr":
                self._stream = cv.cvtColor(self._stream, cv.COLOR_BGR2HSV)
            elif previous.lower() == "rgb":
                self._stream = cv.cvtColor(self._stream, cv.COLOR_RGB2HSV)
            else:
                raise ValueError("Sistema de cor não suportado para conversão para HSV.")
        return self

    def color_segment(self, threshold):
        """Create binary mask based on color threshold."""
        return

    def morph(self, operation, ksize, kshape):
        """Apply morphological operations."""
        if operation == 'closing':
            return self._input
        return

    def log(self, kernel_size):
        kernel= cv.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))


        return self

    def bilateral_filter(self, d=9, sigmaColor=75, sigmaSpace=75):
        """
        Apply bilateral filter for noise reduction while preserving edges.

        Args:
            d: Diameter of pixel neighborhood
            sigmaColor: Filter sigma in color space
            sigmaSpace: Filter sigma in coordinate space
        """
        if self._stream is not None:
            self._stream = cv.bilateralFilter(self._stream, d, sigmaColor, sigmaSpace)
        return self

    def apply_median_filter(self, kernel_size=5):
        """
        Apply median filter for noise reduction.
        Effective for salt-and-pepper noise.
        """
        if self._stream is not None:
            self._stream = cv.medianBlur(self._stream, kernel_size)
        return self

    def clahe(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            clip_limit: Contrast limiting threshold
            tile_grid_size: Size of grid for histogram equalization
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
        """Apply adaptive thresholding to separate light/dark regions."""
        return

    def border_canny(self):
        """Apply Canny edge detection."""
        return

    def surfaceMap(self):
        """Return coordinates of detected planes from binary mask."""
        return

    def robot_eye(self):
        """Combine previous functions for complete image processing."""
        return

    def orb_reveal(self):
        """Implement image descriptor for lighting and obstacle detection."""
        message = ""
        return message
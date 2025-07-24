import numpy as np
import cv2 as cv
from scipy import ndimage, signal
from dataclasses import dataclass
from skimage.measure import shannon_entropy
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

    BasicBinarizeMethod = Literal['histogram', 'otsu', 'adaptive']
    AdvancedBinarizeMethod = Literal['niblack', 'sauvola', 'bernsen']
    HistogramBinarizeMethod = Literal['histo-mean', 'histo-peak']
    TheoreticalBinarizeMethod = Literal['kittler', 'kapur', 'tsai']

    BinarizeMethod = BasicBinarizeMethod | AdvancedBinarizeMethod | HistogramBinarizeMethod | TheoreticalBinarizeMethod

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
        new_photon._processed = np.copy(self._processed)
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
            'rgb': ColorPattern('rgb', 3, cv.COLOR_BGR2RGB),
            'hsv': ColorPattern('hsv', 3, cv.COLOR_BGR2HSV),
            'grayscale': ColorPattern('grayscale', 1, cv.COLOR_BGR2GRAY)
        }

        if isinstance(pattern, ColorPattern):
            return pattern
        elif pattern in patterns:
            return patterns[pattern]
        else:
            raise ValueError(f"Padrão de cores não suportado: {pattern}")

    def calculate_brightness(self, use_purity: bool = False) -> float:

        img = self._purity if use_purity else self._processed
        if img is None:
            raise ValueError("No image available")

        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return np.mean(img)

    def calculate_contrast(self, use_purity: bool = False) -> float:

        img = self._purity if use_purity else self._processed
        if img is None:
            raise ValueError("No image available")

        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return np.std(img)

    def calculate_entropy(self, use_purity: bool = False) -> float:

        img = self._purity if use_purity else self._processed
        if img is None:
            raise ValueError("No image available")

        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return shannon_entropy(img)


    def show_side_by_side(self,
                     title1: str = "Original",
                     title2: str = "Processed",
                     save_path: Optional[str] = None) -> None:

        if self._purity is None or self._processed is None:
            raise ValueError("Both original and processed images must be available")

        fig = plt.figure(figsize=(12, 6))

        # Original image (convert BGR to RGB for matplotlib)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(cv.cvtColor(self._purity, cv.COLOR_BGR2RGB))
        ax1.set_title(title1)
        ax1.axis('off')

        # Processed image (handle grayscale case)
        ax2 = fig.add_subplot(1, 2, 2)
        if len(self._processed.shape) == 2:  # Grayscale
            ax2.imshow(self._processed, cmap='gray')
        else:  # Color
            ax2.imshow(cv.cvtColor(self._processed, cv.COLOR_BGR2RGB))
        ax2.set_title(title2)
        ax2.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Comparison saved to: {save_path}")
            plt.close(fig)
        else:
            plt.show()

    #método de isolamento da ROI
    def readROI(self, initpoint: tuple[int, int], endpoint: tuple [int,int]):
        x1, y1 = initpoint
        x2, y2 = endpoint
        imageRGB= cv.cvtColor(self._purity, cv.COLOR_BGR2RGB)
        height, width = imageRGB.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        region = imageRGB[y1:y2, x1:x2]

        if region.size == 0:
            print('a área selecionada está vazia ou fora da imagem')
            return

        self._processed= region

        return self

    # Métodos de pré-processamento
    def clahe(self, clip_limit=2.0, tile_grid=(8, 8)):
        if self._processed is not None:
            gray = cv.cvtColor(self._processed, cv.COLOR_BGR2GRAY) if len(self._processed.shape) == 3 else self._processed
            clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize= tile_grid)
            self._processed = clahe.apply(gray)
        return self

    def bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        if self._processed is not None:
            self._processed = cv.bilateralFilter(self._processed, d, sigma_color, sigma_space)
        return self

    def apply_median_filter(self, kernel_size=5):
        if self._processed is not None:
            self._processed = cv.medianBlur(self._processed, kernel_size)
        return self

    def morph(self, operation, ksize=3, kshape='rect'):
        if self._processed is not None:
            shape_map = {'rect': cv.MORPH_RECT, 'ellipse': cv.MORPH_ELLIPSE, 'cross': cv.MORPH_CROSS}
            kernel = cv.getStructuringElement(shape_map[kshape], (ksize, ksize))

            if operation == 'open':
                self._processed = cv.morphologyEx(self._processed, cv.MORPH_OPEN, kernel)
            elif operation == 'close':
                self._processed = cv.morphologyEx(self._processed, cv.MORPH_CLOSE, kernel)
        return self

    def binarize(
        self,
        method: BinarizeMethod = 'otsu',
        threshold: Optional[float] = None,
        window_size: int = 11,
        k: float = 0.2,
        r: float = 128
    ) -> 'Photon':
        """
        Apply binarization to the processed image using specified method.
        """
        if self._processed_image is None:
            raise ValueError("No image available for processing")

        if method in ['niblack', 'sauvola', 'bernsen']:
            return self._advanced_binarize(method, window_size, k, r)
        return self._basic_binarize(method, threshold)

    def _basic_binarize(
        self,
        method: Union[BasicBinarizeMethod, HistogramBinarizeMethod, TheoreticalBinarizeMethod],
        threshold: Optional[float] = None
    ) -> 'Photon':
        gray = cv.cvtColor(self._processed_image, cv.COLOR_BGR2GRAY) if len(self._processed_image.shape) == 3 else self._processed_image

        match method:
            case 'histogram':
                thresh = threshold if threshold is not None else np.mean(gray)
                _, self._processed_image = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)

            case 'otsu':
                _, self._processed_image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

            case 'adaptive':
                block_size = 11
                const = threshold if threshold is not None else 2
                self._processed_image = cv.adaptiveThreshold(
                    gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv.THRESH_BINARY, block_size, const
                )

            case 'histo-mean':
                hist = cv.calcHist([gray], [0], None, [256], [0, 256])
                self._processed_image = cv.threshold(gray, np.mean(hist), 255, cv.THRESH_BINARY)[1]

            case 'histo-peak':
                hist = cv.calcHist([gray], [0], None, [256], [0, 256]).flatten()
                peaks, _ = signal.find_peaks(hist, distance=50)
                if len(peaks) >= 2:
                    valley = np.argmin(hist[peaks[0]:peaks[1]]) + peaks[0]
                    self._processed_image = cv.threshold(gray, valley, 255, cv.THRESH_BINARY)[1]
                else:
                    self._processed_image = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)[1]

            case 'kittler':
                def kittler_threshold(img):
                    hist = cv.calcHist([img], [0], None, [256], [0, 256]).flatten()
                    norm_hist = hist / hist.sum()
                    cumsum = np.cumsum(norm_hist)
                    cumsum2 = np.cumsum(norm_hist * np.arange(256))
                    total_mean = cumsum2[-1]

                    best_thresh, min_error = 0, float('inf')

                    for t in range(1, 255):
                        w0, w1 = cumsum[t], 1 - cumsum[t]
                        if w0 == 0 or w1 == 0: continue

                        mean0 = cumsum2[t] / w0
                        mean1 = (total_mean - cumsum2[t]) / w1
                        var0 = np.sum(norm_hist[:t] * (np.arange(t) - mean0)**2) / w0
                        var1 = np.sum(norm_hist[t:] * (np.arange(t, 256) - mean1)**2) / w1

                        error = 1 + 2*(w0*np.log(var0) + w1*np.log(var1)) - 2*(w0*np.log(w0) + w1*np.log(w1))
                        if error < min_error:
                            min_error, best_thresh = error, t
                    return best_thresh

                thresh = kittler_threshold(gray)
                self._processed_image = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)[1]

            case 'kapur':
                def kapur_threshold(img):
                    hist = cv.calcHist([img], [0], None, [256], [0, 256]).flatten()
                    norm_hist = hist / hist.sum()
                    cumsum = np.cumsum(norm_hist)

                    best_thresh, max_entropy = 0, -float('inf')

                    for t in range(1, 255):
                        w0, w1 = cumsum[t], 1 - cumsum[t]
                        if w0 == 0 or w1 == 0: continue

                        entropy0 = -np.sum((norm_hist[:t]/w0) * np.log(norm_hist[:t]/w0 + 1e-10))
                        entropy1 = -np.sum((norm_hist[t:]/w1) * np.log(norm_hist[t:]/w1 + 1e-10))
                        total_entropy = entropy0 + entropy1

                        if total_entropy > max_entropy:
                            max_entropy, best_thresh = total_entropy, t
                    return best_thresh

                thresh = kapur_threshold(gray)
                self._processed_image = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)[1]

            case 'tsai':
                def tsai_threshold(img):
                    hist = cv.calcHist([img], [0], None, [256], [0, 256]).flatten()
                    norm_hist = hist / hist.sum()
                    cumsum = np.cumsum(norm_hist)
                    cumsum2 = np.cumsum(norm_hist * np.arange(256))
                    total_mean = cumsum2[-1]

                    best_thresh, min_diff = 0, float('inf')

                    for t in range(1, 255):
                        w0, w1 = cumsum[t], 1 - cumsum[t]
                        if w0 == 0 or w1 == 0: continue

                        mean0 = cumsum2[t] / w0
                        mean1 = (total_mean - cumsum2[t]) / w1
                        var0 = np.sum(norm_hist[:t] * (np.arange(t) - mean0)**2) / w0
                        var1 = np.sum(norm_hist[t:] * (np.arange(t, 256) - mean1)**2) / w1

                        diff = abs(var0 + var1 - (w0*var0 + w1*var1 + w0*w1*(mean0 - mean1)**2))
                        if diff < min_diff:
                            min_diff, best_thresh = diff, t
                    return best_thresh

                thresh = tsai_threshold(gray)
                self._processed_image = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)[1]

            case 'mobile-gauss':
                window_size = 11 if window_size % 2 == 0 else window_size
                self._processed_image = cv.adaptiveThreshold(
                    gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv.THRESH_BINARY, window_size, threshold if threshold is not None else 2
                )

            case _:
                raise ValueError(f"Invalid binarization method: {method}")

        return self

    def _advanced_binarize(
        self,
        method: AdvancedBinarizeMethod,
        window_size: int = 11,
        k: float = 0.2,
        r: float = 128
    ) -> 'Photon':
        gray = cv.cvtColor(self._processed_image, cv.COLOR_BGR2GRAY) if len(self._processed_image.shape) == 3 else self._processed_image

        match method:
            case 'niblack':
                if window_size % 2 == 0:
                    window_size += 1
                mean = cv.boxFilter(gray, cv.CV_32F, (window_size, window_size), normalize=True)
                stddev = cv.boxFilter(gray**2, cv.CV_32F, (window_size, window_size), normalize=True)
                stddev = np.sqrt(stddev - mean**2)
                threshold_niblack = mean + k * stddev
                self._processed_image = np.where(gray > threshold_niblack, 255, 0).astype(np.uint8)

            case 'sauvola':
                if window_size % 2 == 0:
                    window_size += 1
                mean = cv.boxFilter(gray, cv.CV_32F, (window_size, window_size), normalize=True)
                stddev = cv.boxFilter(gray**2, cv.CV_32F, (window_size, window_size), normalize=True)
                stddev = np.sqrt(stddev - mean**2)
                threshold_sauvola = mean * (1 + k * (stddev / r - 1))
                self._processed_image = np.where(gray > threshold_sauvola, 255, 0).astype(np.uint8)

            case 'bernsen':
                min_img = cv.erode(gray, np.ones((window_size, window_size), np.uint8))
                max_img = cv.dilate(gray, np.ones((window_size, window_size), np.uint8))
                contrast = max_img - min_img
                threshold_bernsen = (min_img + max_img) / 2
                self._processed_image = np.where(
                    (contrast > 15) & (gray > threshold_bernsen),
                    255, 0
                ).astype(np.uint8)

            case _:
                raise ValueError(f"Invalid binarization method: {method}")

        return self
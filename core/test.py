import os
import cv2
from photon import Photon

# Configuração
OUTPUT_DIR = "test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carregar imagens de teste
print("Carregando imagens de teste...")
try:
    images = {
        'tangela': Photon.pokefetch('tangela'),
        'tentacruel': Photon.pokefetch('tentacruel'),
        'terastodon': Photon.magic_gather('terastodon'),
        'random_poke': Photon.pokefetch(),
        'random_card': Photon.magic_gather()
    }
    print("Imagens carregadas com sucesso!")
except Exception as e:
    print(f"Erro ao carregar imagens: {e}")
    exit()

# Funções de teste
def test_clahe():
    print("\nTestando CLAHE...")
    img = images['tangela'].copy()
    result = img.clahe(clip_limit=2.0)
    img.show_side_by_side(result, save_path=os.path.join(OUTPUT_DIR, 'clahe_result.png'))

def test_filters():
    print("\nTestando Filtros...")
    img = images['tentacruel'].copy()
    result = img.bilateral_filter().apply_median_filter()
    img.show_side_by_side(result, save_path=os.path.join(OUTPUT_DIR, 'filters_result.png'))

def test_morphology():
    print("\nTestando Morfologia...")
    img = images['random_poke'].copy()
    opened = img.copy().morph('open', ksize=5)
    closed = img.copy().morph('close', ksize=3)

    img.show_side_by_side(opened, title2="Abertura", save_path=os.path.join(OUTPUT_DIR, 'morph_open.png'))
    img.show_side_by_side(closed, title2="Fechamento", save_path=os.path.join(OUTPUT_DIR, 'morph_close.png'))

def test_binarization():
    print("\nTestando Binarização...")
    img = images['terastodon'].copy()

    methods = [
        ('histogram', None),
        ('histogram', 100),
        ('otsu', None),
        ('adaptive', None),
        ('adaptive', 5)
    ]

    for i, (method, thresh) in enumerate(methods):
        try:
            result = img.copy().binarize(method=method, threshold=thresh)
            fname = f"bin_{method}_{i}.png"
            img.show_side_by_side(result, title2=f"{method}{f' ({thresh})' if thresh else ''}",
                                save_path=os.path.join(OUTPUT_DIR, fname))
        except Exception as e:
            print(f"Erro em {method}: {e}")

# Executar testes
if __name__ == "__main__":
    print("\nIniciando testes de pré-processamento...")
    test_clahe()
    test_filters()
    test_morphology()
    test_binarization()
    print(f"\nTodos os testes completados! Resultados salvos em: {os.path.abspath(OUTPUT_DIR)}")
from photon import Photon
from test import *
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import shannon_entropy
import csv

def calcular_brilho(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(img)


def calcular_contraste(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.std(img)


def calcular_entropia(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return shannon_entropy(img)


def avaliar_melhoria(delta_contraste, delta_entropia, contraste_min=5, entropia_min=0.2):
    return delta_contraste > contraste_min and delta_entropia > entropia_min


def aplicar_filtro_por_nome(photon: Photon, nome_filtro: str) -> Photon:
    filtrada = photon.copy()
    if nome_filtro == "clahe":
        filtrada.clahe()
    elif nome_filtro == "binarize_histogram":
        filtrada.binarize("histogram")
    elif nome_filtro == "binarize_otsu":
        filtrada.binarize("otsu")
    elif nome_filtro == "apply_median_filter":
        filtrada.apply_median_filter()
    elif nome_filtro == "morph_open":
        filtrada.morph("open")
    elif nome_filtro == "bilateral_filter":
        filtrada.bilateral_filter()
    elif nome_filtro == "clahe+bilateral":
        filtrada.clahe().bilateral_filter()
    else:
        raise ValueError(f"Filtro desconhecido: {nome_filtro}")
    return filtrada


def testar_filtro_completo(filtro_nome, qtd=50, image_type="art_crop"):
    resultados = []
    melhorou = 0

    for i in range(qtd):
        try:
            print(f"[{i+1:02d}/{qtd}] Testando filtro '{filtro_nome}'...")
            original = Photon.magic_gather(image_type=image_type)
            processada = aplicar_filtro_por_nome(original, filtro_nome)

            img1 = original._stream
            img2 = processada._stream

            b1, b2 = calcular_brilho(img1), calcular_brilho(img2)
            c1, c2 = calcular_contraste(img1), calcular_contraste(img2)
            e1, e2 = calcular_entropia(img1), calcular_entropia(img2)

            delta_b, delta_c, delta_e = b2 - b1, c2 - c1, e2 - e1
            sucesso = avaliar_melhoria(delta_c, delta_e)
            if sucesso:
                melhorou += 1

            resultados.append([i+1, b1, b2, delta_b, c1, c2, delta_c, e1, e2, delta_e, sucesso])

        except Exception as e:
            print(f"[ERRO] {e}")

    print(f"\n✅ {melhorou}/{qtd} imagens melhoraram com o filtro '{filtro_nome}'.")

    nome_csv = f"resultados_metricas_{filtro_nome}.csv"
    with open(nome_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Brilho_antes", "Brilho_depois", "Delta_brilho",
                         "Contraste_antes", "Contraste_depois", "Delta_contraste",
                         "Entropia_antes", "Entropia_depois", "Delta_entropia", "Melhorou"])
        writer.writerows(resultados)

    ids = [r[0] for r in resultados]
    deltas_c = [r[6] for r in resultados]
    deltas_e = [r[9] for r in resultados]

    plt.figure(figsize=(10, 5))
    plt.plot(ids, deltas_c, label="Δ Contraste", marker='o')
    plt.plot(ids, deltas_e, label="Δ Entropia", marker='x')
    plt.axhline(0, color="gray", linestyle="--")
    plt.title(f"Métricas para filtro: {filtro_nome}")
    plt.xlabel("Imagem")
    plt.ylabel("Variação")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"grafico_metricas_{filtro_nome}.png")
    plt.show()


def test_image_pipeline(image_path):
    """Demonstra o encadeamento de métodos em uma imagem estática."""
    print("-- Teste de pipeline em imagem estática --")
    try:
        processed_photon = (Photon.from_path(image_path)
                              .clahe()
                              .bilateral_filter()
                              .orb_reveal())

        processed_photon.show("Resultado do Pipeline ORB")

    except (FileNotFoundError, TypeError) as e:
        print(f"Erro: {e}")

def test_surface_map(image_path):
    """Demonstra a detecção de superfícies."""
    print("\n-- Testando a detecção de superfícies (surfaceMap) --")
    try:
        lower_green = [35, 40, 40]
        upper_green = [85, 255, 255]

        mask = (Photon.from_path(image_path)
                      .to_hsv()
                      .color_segment(lower_green, upper_green)
                      .closing((10,10))) 

        contours, image_with_surfaces = Photon.surfaceMap(mask.image)
        print(f"SurfaceMap encontrou {len(contours)} contornos.")

        cv2.imshow("Mascara usada", mask.image)
        cv2.imshow("Superfícies Detectadas", image_with_surfaces)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except (FileNotFoundError, TypeError) as e:
        print(f"Erro: {e}")

def run_realtime_robot_eye():
    """Executa a função robot_eye em tempo real com a webcam."""
    print("\n-- Executando RobotEye em tempo real (pressione 'q' para sair) --")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame.")
            break

        processed_frame = Photon.robot_eye(frame)

        cv2.imshow("RobotEye - Real-Time", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def menu():
    while True:
        print("\n=== MENU FONTE DE IMAGEM ===")
        print("1 - Mostrar carta do Magic específica")
        print("2 - Mostrar carta aleatória do Magic")
        print("3 - Mostrar Pokémon específico")
        print("4 - Mostrar Pokémon aleatório")
        print("0 - Sair")

        escolha_fonte = input("Escolha uma opção: ")

        try:
            if escolha_fonte == "0":
                print("Encerrando...")
                break

            if escolha_fonte == "1":
                nome = input("Nome da carta: ")
                print("\nEscolha o tipo de imagem:")
                print("1 - Carta completa (com moldura e texto)")
                print("2 - Apenas a arte (sem moldura)")
                tipo = input("Escolha: ")
                image_type = "normal" if tipo == "1" else "art_crop"
                imagem = Photon.magic_gather(nome, image_type=image_type)

            elif escolha_fonte == "2":
                print("\nEscolha o tipo de imagem:")
                print("1 - Carta completa (com moldura e texto)")
                print("2 - Apenas a arte (sem moldura)")
                tipo = input("Escolha: ")
                image_type = "normal" if tipo == "1" else "art_crop"
                imagem = Photon.magic_gather(image_type=image_type)

            elif escolha_fonte == "3":
                nome = input("Nome do Pokémon: ")
                imagem = Photon.pokefetch(nome)

            elif escolha_fonte == "4":
                imagem = Photon.pokefetch()

            else:
                print("Opção inválida.")
                continue

            while True:
                print("\n=== MENU AÇÃO COM A IMAGEM ===")
                print("1 - Exibir imagem original")
                print("2 - Comparar Binarização Otsu (Figura 2)")
                print("3 - Comparar Mediana vs Bilateral (Figura 3)")
                print("4 - Comparar CLAHE + filtros (completo)")
                print("5 - Comparar apenas Mediana")
                print("6 - Comparar apenas Bilateral")
                print("7 - Comparar morfologia 'open' (Figura 4)")
                print("0 - Voltar")

                escolha_acao = input("Escolha uma ação: ")

                if escolha_acao == "0":
                    break

                elif escolha_acao == "1":
                    imagem.show("Imagem Original")

                elif escolha_acao == "2":
                    bin_otsu = Photon(imagem._stream.copy()).binarize("otsu")
                    imagem.show_side_by_side(bin_otsu, "Original", "Binarização Otsu", save_path="figura2_otsu.png")
                    print("Figura 2: Binarização de Otsu salva como figura2_otsu.png")

                elif escolha_acao == "3":
                    img_median = Photon(imagem._stream.copy()).apply_median_filter()
                    img_bilateral = Photon(imagem._stream.copy()).bilateral_filter()
                    img_median.show_side_by_side(img_bilateral, "Filtro Mediana", "Filtro Bilateral", save_path="figura3_ruido.png")
                    print("Figura 3: Comparação de filtros de ruído salva como figura3_ruido.png")

                elif escolha_acao == "4":
                    img_tudo = Photon(imagem._stream.copy()).clahe().apply_median_filter().bilateral_filter()
                    imagem.show_side_by_side(img_tudo, "Original", "CLAHE + Filtros", save_path="figura_completa.png")
                    print("Comparação completa salva como figura_completa.png")

                elif escolha_acao == "5":
                    img_median = Photon(imagem._stream.copy()).apply_median_filter()
                    imagem.show_side_by_side(img_median, "Original", "Filtro Mediana", save_path="figura_mediana.png")
                    print("Filtro de Mediana salvo como figura_mediana.png")

                elif escolha_acao == "6":
                    img_bilateral = Photon(imagem._stream.copy()).bilateral_filter()
                    imagem.show_side_by_side(img_bilateral, "Original", "Filtro Bilateral", save_path="figura_bilateral.png")
                    print("Filtro Bilateral salvo como figura_bilateral.png")

                elif escolha_acao == "7":
                    bin_otsu = Photon(imagem._stream.copy()).binarize("otsu")
                    morf = Photon(bin_otsu._stream.copy()).morph("open")
                    bin_otsu.show_side_by_side(morf, "Binária com ruído", "Após Abertura", save_path="figura4_morfologia.png")
                    print("Figura 4: Comparação morfológica salva como figura4_morfologia.png")

                else:
                    print("Ação inválida.")

        except Exception as e:
            print(f"Erro: {e}")



if __name__ == "__main__":
   menu()
    #menu()

    # Carrega uma carta específica de Magic: The Gathering usando a API da Scryfall.
    # photon = Photon.from_magic_card("Black Lotus")

    # Exibe a imagem da carta carregada em uma janela com o título "Carta Black Lotus".
    # photon.show("Carta Black Lotus")


    # Carrega uma carta aleatória de Magic: The Gathering usando a API da Scryfall.
    # photon = Photon.from_random_magic_card()

    # Exibe a imagem da carta aleatória em uma janela com o título "Carta Aleatória".
    # photon.show("Carta Aleatoria")

    # Carrega a imagem oficial do Pokémon "Pikachu" usando a PokéAPI.
    # pokemon = Photon.from_pokemon("Pikachu")

    # Exibe a imagem do Pikachu em uma janela com o título correspondente.
    # pokemon.show("Pikachu")

    # pikachu = Photon.from_pokemon("Pikachu")
    # clahe_pikachu = Photon(pikachu._stream.copy()).clahe()

    # pikachu.show_side_by_side(clahe_pikachu, "Original", "Com CLAHE")


    # Carrega a imagem oficial de um Pokémon aleatório usando a PokéAPI.
    # O Pokémon é escolhido de forma aleatória com base no seu ID (1 a 898).
    # pokemon = Photon.from_random_pokemon()

    # Exibe a imagem do Pokémon aleatório em uma janela com o título "Pokemon Aleatorio".
    # pokemon.show("Pokemon Aleatorio")

    # TESTES

    # TESTE 1: o pipeline de filtros + ORB
    # test_image_pipeline(IMAGE_PATH)

    # TESTE 2: a segmentação por cor e detecção de superfície
    # test_surface_map(IMAGE_PATH)

    # TESTE 3:  o modo de câmera em tempo real
    # run_realtime_robot_eye()
from photon import Photon
from test import *
import cv2

def test_image_pipeline(image_path):
    """Demonstra o encadeamento de métodos em uma imagem estática."""
    print("-- Teste de pipeline em imagem estática --")
    try:
        # Carrega a imagem e aplica um pipeline
        processed_photon = (Photon.from_path(image_path)
                              .clahe()
                              .bilateral_filter()
                              .orb_reveal())

        # Mostra o resultado final
        processed_photon.show("Resultado do Pipeline ORB")

    except (FileNotFoundError, TypeError) as e:
        print(f"Erro: {e}")

def test_surface_map(image_path):
    """Demonstra a detecção de superfícies."""
    print("\n-- Testando a detecção de superfícies (surfaceMap) --")
    try:
        # Cria uma máscara binária
        # Segmentar uma cor
        # Exemplo para segmentar algo verde
        lower_green = [35, 40, 40]
        upper_green = [85, 255, 255]

        mask = (Photon.from_path(image_path)
                      .to_hsv()
                      .color_segment(lower_green, upper_green)
                      .closing((10,10))) # Fecha buracos na máscara

        #  surfaceMap na máscara
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
    cap = cv2.VideoCapture(0) # 0 para a webcam padrão

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame.")
            break

        # Aplica o pipeline do robot_eye
        processed_frame = Photon.robot_eye(frame)

        # Exibe o resultado
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

            # Etapa 1: Carrega imagem
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

            # Etapa 2: O que fazer com a imagem?
            while True:
                print("\n=== MENU AÇÃO COM A IMAGEM ===")
                print("1 - Exibir imagem original")
                print("2 - Aplicar CLAHE e comparar")
                print("3 - Aplicar filtros de ruído (mediana + bilateral) e comparar")
                print("4 - Aplicar CLAHE + filtros e comparar")
                print("5 - Aplicar apenas filtro de mediana e comparar")
                print("6 - Aplicar apenas filtro bilateral e comparar")
                print("0 - Voltar para o menu anterior")

                escolha_acao = input("Escolha uma ação: ")

                if escolha_acao == "0":
                    break

                elif escolha_acao == "1":
                    imagem.show("Imagem Original")

                elif escolha_acao == "2":
                    img_clahe = Photon(imagem._stream.copy()).clahe()
                    imagem.show_side_by_side(img_clahe, "Original", "Com CLAHE")

                elif escolha_acao == "3":
                    img_filtrada = Photon(imagem._stream.copy()).apply_median_filter().bilateral_filter()
                    imagem.show_side_by_side(img_filtrada, "Original", "Pós-Filtros")

                elif escolha_acao == "4":
                    img_tudo = Photon(imagem._stream.copy()).clahe().apply_median_filter().bilateral_filter()
                    imagem.show_side_by_side(img_tudo, "Original", "CLAHE + Filtros")

                elif escolha_acao == "5":
                    img_median = Photon(imagem._stream.copy()).apply_median_filter()
                    imagem.show_side_by_side(img_median, "Original", "Filtro de Mediana")

                elif escolha_acao == "6":
                    img_bilateral = Photon(imagem._stream.copy()).bilateral_filter()
                    imagem.show_side_by_side(img_bilateral, "Original", "Filtro Bilateral")

                else:
                    print("Ação inválida.")
        except Exception as e:
            print(f"Erro: {e}")


if __name__ == "__main__":
    IMAGE_PATH = ".jpg"
    print("\nStarting tests...")
    test_clahe()
    test_bilateral_median()
    test_morph_operations()
    test_individual_binarization()
    print("\nAll tests completed!")
    plt.show()
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
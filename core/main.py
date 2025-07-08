from photon import Photon
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


if __name__ == "__main__":
    IMAGE_PATH = ".jpg"

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
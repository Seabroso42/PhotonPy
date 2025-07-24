<div align="left">
  <img src="./interface/assets/readmeLogo.png" alt="Project' Logo and name, snake with a lamp pixel art next to an also pixelated text " width="480" height="210" >
</div>
A Wrapper on some computer vision methods in order to precisely track light and darkness in a image or video.

## Problem to be Addressed:
High complexity in converting images and videos into machine-readable data, in an open-source manner.

## Objectives
To abstract the complexity of analyzing videos and images for training machines to recognize planes and lighting levels. 
(AI or inverse kinematics for movement in robotics)
To seek harmony between classic techniques such as Laplacian, Canny, Sobel, Median, Bilateral filters, etc. 
In order to create a function called “robotEye()” that abstracts the connection with devices and has an algorithm useful in several applications.

## Specific Objectives:

- Measurement of brightness levels with fuzzy
- Surface recognition and mapping
- Encapsulation of pre- and post-processing methods of images
- Simplification of results in a simple GUI
- Application of linear algebra formulas for image processing (future implementation).
- Integration with cameras and application of filters in real time. (final objective)

## Filters used:
- Median Filter (noise reduction)
- Bilateral Filter (max. quality)
- Morphological: Opening and Closing.

## Fluxo do Projeto

Aqui está uma visão geral da arquitetura e do fluxo de dados do projeto:

```mermaid
graph TD
    A[Início: Usuário executa main.py] --> B{Escolha da Fonte de Imagem};
    B --> C[API: pokefetch / magic_gather];
    B --> D[Arquivo Local];
    B --> E[Câmera ao vivo];
    
    subgraph "Camada de Aquisição (RobotEye / Photon)"
        C --> F[Criação do Objeto Photon];
        D --> F;
        E --> F;
    end

    subgraph "Motor de Processamento (Photon)"
        F --> G{Pipeline de Processamento};
        G --> H["Método 1 (.clahe)"];
        H --> I["Método 2 (.binarize)"];
        I --> J["etc..."];
    end

    J --> K[Objeto Photon com Imagem Processada];

    subgraph "Camada de Saída"
        K --> L[Saída Visual (.show_side_by_side)];
        K --> M[Saída de Dados (Gráficos, CSV)];
    end

    K --> N[Futuro: ModelRunner (IA)];

## License

[MIT](https://choosealicense.com/licenses/mit/)

import os
import numpy as np
import cv2 as cv
import customtkinter as ctk
from PIL import Image
from tkinter import filedialog
import requests
import random
from core.photon import Photon 

current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, "assets/CameraEnhance.png")

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.geometry("1000x600")
app.title("Photon-Py")

left_panel = ctk.CTkFrame(app, width=300, height=500, corner_radius=10)
left_panel.place(x=20, y=20)

tab_view = ctk.CTkTabview(left_panel, width=280, height=220)
tab_view.pack(pady=(40, 10))
tab2 = tab_view.add("Imagem")

imagem_label = None
btn_remover = None
btn_adicionar_imagem = None
photon_instance = None


def remover_imagem():
    global imagem_label, btn_remover, photon_instance
    if imagem_label:
        imagem_label.destroy()
    if btn_remover:
        btn_remover.destroy()
    photon_instance = None
    btn_adicionar_imagem.place(relx=0.5, rely=0.5, anchor="center")


def adicionar_imagem_cv2(imagem_cv):
    global imagem_label, btn_remover, photon_instance

    if imagem_cv.shape[2] == 4:
        imagem_cv = cv.cvtColor(imagem_cv, cv.COLOR_BGRA2BGR)

    photon_instance = Photon(imagem_cv)  

    imagem_rgb = cv.cvtColor(imagem_cv, cv.COLOR_BGR2RGB)
    imagem_pil = Image.fromarray(imagem_rgb).resize((260, 180))
    imagem_ctk = ctk.CTkImage(light_image=imagem_pil, size=(260, 180))

    if btn_adicionar_imagem:
        btn_adicionar_imagem.place_forget()

    imagem_label = ctk.CTkLabel(tab2, image=imagem_ctk, text="")
    imagem_label.place(x=10, y=10)

    btn_remover = ctk.CTkButton(
        master=left_panel,
        text="✕", width=30, height=30,
        font=("Arial", 16),
        fg_color="red",
        text_color="white",
        corner_radius=50,
        command=remover_imagem
    )
    btn_remover.place(x=230, y=10)

def carregar_imagem_local():
    caminho = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.gif")])
    if caminho:
        img = cv.imread(caminho, cv.IMREAD_COLOR)
        adicionar_imagem_cv2(img)
        log_box.insert("end", f"\n[INFO] Imagem carregada: {caminho}")


btn_adicionar_imagem = ctk.CTkButton(
    tab2, text="+", width=50, height=50, font=("Arial", 28),
    command=carregar_imagem_local,
    fg_color="#F4A300", text_color="black"
)
btn_adicionar_imagem.place(relx=0.5, rely=0.5, anchor="center")


def aplicar_algoritmo(tipo):
    global photon_instance
    if not photon_instance:
        return
    log_box.insert("end", f"\n[INFO] Aplicando algoritmo: {tipo}")
    result = photon_instance.copy()

    if tipo == "clahe":
        result.clahe()
    elif tipo == "binarize_histogram":
        result.binarize("histogram")
    elif tipo == "binarize_otsu":
        result.binarize("otsu")
    elif tipo == "apply_median_filter":
        result.apply_median_filter()
    elif tipo == "morph_open":
        result.morph("open")
    elif tipo == "bilateral_filter":
        result.bilateral_filter()

    imagem_cv = result._stream
    if len(imagem_cv.shape) == 2 or imagem_cv.shape[2] == 1:  
        imagem_cv = cv.cvtColor(imagem_cv, cv.COLOR_GRAY2RGB)
    elif imagem_cv.shape[2] == 4:
        imagem_cv = cv.cvtColor(imagem_cv, cv.COLOR_BGRA2RGB)
    else:
        imagem_cv = cv.cvtColor(imagem_cv, cv.COLOR_BGR2RGB)

        

    pil_img = Image.fromarray(imagem_cv).resize((580, 330))
    img_ctk = ctk.CTkImage(light_image=pil_img, size=(580, 330))

    if hasattr(app, "imagem_label_output"):
        app.imagem_label_output.destroy()

    app.imagem_label_output = ctk.CTkLabel(output_panel, image=img_ctk, text="")
    app.imagem_label_output.place(x=10, y=10)


def abrir_menu_algoritmo():
    if not photon_instance:
        log_box.insert("end", "\n[ERRO] Nenhuma imagem carregada.")
        return

    menu = ctk.CTkToplevel(app)
    menu.title("Selecionar Algoritmo")
    menu.geometry("300x260")
    menu.grab_set()

    tipo = ctk.StringVar(value="clahe")
    ctk.CTkLabel(menu, text="Escolha um algoritmo:").pack(pady=10)
    for opt in ["clahe", "binarize_histogram", "binarize_otsu", "apply_median_filter", "morph_open", "bilateral_filter"]:
        ctk.CTkRadioButton(menu, text=opt, variable=tipo, value=opt).pack(anchor="w", padx=20)

    ctk.CTkButton(menu, text="Aplicar", command=lambda: (aplicar_algoritmo(tipo.get()), menu.destroy())).pack(pady=15)


def importar_api():
    menu = ctk.CTkToplevel(app)
    menu.title("Importar Imagem da API")
    menu.geometry("340x320")
    menu.grab_set()

    fonte = ctk.StringVar(value="magic")
    tipo_imagem = ctk.StringVar(value="art_crop")
    nome_entry = ctk.CTkEntry(menu, placeholder_text="Nome (opcional)")
    nome_entry.pack(pady=10)

    ctk.CTkLabel(menu, text="Fonte da imagem:").pack()
    ctk.CTkRadioButton(menu, text="Magic", variable=fonte, value="magic").pack(anchor="w", padx=30)
    ctk.CTkRadioButton(menu, text="Pokémon", variable=fonte, value="pokemon").pack(anchor="w", padx=30)

    ctk.CTkLabel(menu, text="Tipo (Magic):").pack(pady=(10, 0))
    ctk.CTkRadioButton(menu, text="Arte", variable=tipo_imagem, value="art_crop").pack(anchor="w", padx=30)
    ctk.CTkRadioButton(menu, text="Completa", variable=tipo_imagem, value="normal").pack(anchor="w", padx=30)

    def importar():
        nome = nome_entry.get().strip()
        try:
            if fonte.get() == "magic":
                imagem = Photon.magic_gather(nome or None, image_type=tipo_imagem.get())
                log_box.insert("end", f"\n[INFO] Magic: {nome or 'Aleatório'}")
            else:
                imagem = Photon.pokefetch(nome or None)
                log_box.insert("end", f"\n[INFO] Pokémon: {nome or 'Aleatório'}")

            adicionar_imagem_cv2(imagem._stream)
            menu.destroy()
        except Exception as e:
            log_box.insert("end", f"\n[ERRO] {str(e)}")

    ctk.CTkButton(menu, text="Importar", command=importar).pack(pady=20)


ctk.CTkButton(left_panel, text="Aplicar Algoritmo", command=abrir_menu_algoritmo,
              fg_color="#F4A300", text_color="black").pack(pady=10)

ctk.CTkButton(left_panel, text="Importar Imagem da API", command=importar_api,
              fg_color="#F4A300", text_color="black").pack(pady=10)

output_panel = ctk.CTkFrame(app, width=600, height=350, fg_color="#EB8F0C", corner_radius=10)
output_panel.place(x=350, y=20)

icon_img = ctk.CTkImage(light_image=Image.open(image_path), size=(60, 45))
ctk.CTkLabel(app, image=icon_img, text="", width=60, height=35, fg_color="#EB8F0C", corner_radius=8)\
    .place(x=620, y=0)

log_box = ctk.CTkTextbox(app, width=600, height=120)
log_box.place(x=350, y=400)
log_box.insert("0.0", "")

ctk.CTkLabel(app, text="🐍 PHOTON-PY", font=("Consolas", 20, "bold")).place(x=20, y=0)

app.mainloop()

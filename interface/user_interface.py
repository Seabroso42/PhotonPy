#fazer uma interface gráfica básica pra mostrar os resultados na aula
import customtkinter as customtk
class PhotonApp(customtk.CTk):

  def __init__(self, master= None):
    super().__init__()
    self.title("Photon-Py")
    self.geometry("1080x640")
    self.grid_columnconfigure(0, weight = 1)
    self.grid_rowconfigure(0, weight = 1)
    main= customtk.CTkFrame()
    main.width()
    main.height
    main.bg_color()



PhotonApp().mainloop()
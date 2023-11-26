import tkinter as tk
from Kmeans import Kmeans


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kmeans Clustering")
        self.geometry("1280x720")
        self.minsize(1280, 720)
        self.maxsize(1280, 720)
        self.play = Kmeans(self)

    def run(self):
        self.mainloop()


app = App()
app.run()

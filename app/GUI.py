from tkinter import *
import matplotlib
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import cv2
import numpy as np


def test():
    print(edtx_add.get())

    fig = Figure()
    img = plt.imread(edtx_add.get())
    hei, wid, c = np.shape(img)
    fig.figimage(img, 0,0)

    root.geometry(str(hei) + "x" + str(wid+50))

    canvas = FigureCanvasTkAgg(fig, master=root) 
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)


root = Tk()
root.title("Fermented cacao bean classifier")
root.geometry("800x600")
root.resizable(False, False)

add_frame = Frame(root)
add_frame.pack(side=TOP)

btn_frame = Frame(root)
btn_frame.pack(side=TOP)

im_add = Label(add_frame, text="Image address: ")
edtx_add = Entry(add_frame)
btn_show_img = Button(btn_frame, text="Show image", command=test)
btn_classify = Button(btn_frame, text="Classify")


im_add.pack(side=LEFT, expand=True)
edtx_add.pack(side=LEFT)

btn_show_img.pack(side=LEFT)
btn_classify.pack(side=LEFT)

pic_frame = Frame(root)
pic_frame.pack(side=TOP)





root.mainloop()
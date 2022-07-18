from tkinter import *
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from image_extractor import feature_extract

classes_name = ['Agglutinated', 'Brittle', 'Compartmentalized_Brown', 'Compartmentalized_PartiallyPurple', 'Compartmentalized_Purple', 'Compartmentalized_Slaty', 'Compartmentalized_White', 'Flattened', 'Moldered', 'Plated_Brown', 'Plated_PartiallyPurple', 'Plated_Purple', 'Plated_Slaty', 'Plated_White']


def test():
    print(edtx_add.get())

    fig = Figure()
    img = cv2.imread(edtx_add.get())
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hei, wid, c = np.shape(img)
    fig.figimage(im_rgb, 0,0)

    root.geometry(str(hei) + "x" + str(wid+50))

    canvas = FigureCanvasTkAgg(fig, master=root) 
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

def predict():
    im = cv2.imread(edtx_add.get())
    print("predicting", np.shape(im))
    extractor.extract(im)
    print(np.shape([extractor.concat()]))
    result = model.predict_on_batch([extractor.concat()])
    print(result)
    print(classes_name[np.argmax(result)])

model = keras.models.load_model('D:./TF_backup/mlp/78_overall_geometry_overall_rgb_color_grid_glcm_grid_comp_hsv_red_haralick_blue_haralick_green_haralick.h5')

extractor = feature_extract()


root = Tk()
root.title("extractorrmented cacao bean classifier")
root.geometry("550x500")
root.resizable(False, False)

add_frame = Frame(root)
add_frame.pack(side=TOP)

btn_frame = Frame(root)
btn_frame.pack(side=TOP)

im_add = Label(add_frame, text="Image address: ")
edtx_add = Entry(add_frame)
btn_show_img = Button(btn_frame, text="Show image", command=test)
btn_classify = Button(btn_frame, text="Classify", command=lambda: predict())


im_add.pack(side=LEFT, expand=True)
edtx_add.pack(side=LEFT)

btn_show_img.pack(side=LEFT)
btn_classify.pack(side=LEFT)

pic_frame = Frame(root)
pic_frame.pack(side=TOP)





root.mainloop()
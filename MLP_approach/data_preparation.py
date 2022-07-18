import multiprocessing
from matplotlib import image
import numpy as np
import pathlib
import cv2
import time
from image_extractor import feature_extract
import sys
import matplotlib.pyplot as plt

# input_shape = (img_width, img_height, 3)
# train_dir = pathlib.Path('D:/Thesis_data/mlp_data/training_img')
# test_dir  = pathlib.Path('D:/Thesis_data/mlp_data/testing_img')
# checkpoint_dir = pathlib.Path('D:./TF_checkpoint/cacao_CNN/weight/')
# model_dir = pathlib.Path('D:./TF_backup/mlp/mlp.h5')
# model_plot_dir = pathlib.Path('D:./TF_backup/mlp/mlp.png')

classes_name = ['Agglutinated', 'Brittle', 'Compartmentalized_Brown', 'Compartmentalized_PartiallyPurple', 'Compartmentalized_Purple', 'Compartmentalized_Slaty', 'Compartmentalized_White', 'Flattened', 'Moldered', 'Plated_Brown', 'Plated_PartiallyPurple', 'Plated_Purple', 'Plated_Slaty', 'Plated_White']
training_address = 'D:/Thesis_data/mlp_data/training_img/'
testing_address = 'D:/Thesis_data/mlp_data/testing_img/'
corrected_address = 'D:/Thesis_data/transfer_learning/corrected/'
original_address = 'D:/Thesis_data/transfer_learning/original/'

def batch_prepare(i):
    extractor = feature_extract()

    overall_geometry = []
    overall_rgb = []
    overall_hsv = []
    n1 = []
    structure = []
    n2 = []
    moldered =[]
    glcm = []
    color_grid = []
    glcm_grid = []
    lbp_hist = []
    haralick = []
    comp_hsv = []
    red_h = []
    blue_h= []
    green_h = []
    
    # t = tmp
    # t[i] = 1
    for j in range(1,91):
        print(classes_name[i],'IMAGE ', j)
        # add = testing_address + classes_name[i]+ '/image(' + str(j) + ').JPG'
        # image = cv2.imread(add)
        # extractor.extract(image)
        # overall_geometry.append(extractor.overall_geometry)
        # overall_rgb.append(extractor.overall_rgb_stat)
        # overall_hsv.append(extractor.overall_hsv_stat)
        # n1.append(extractor.n1)
        # structure.append(extractor.structure)
        # n2.append(extractor.n2)
        # moldered.append(extractor.mold)
        # color_grid.append(extractor.grid_stat)
        # glcm_grid.append(extractor.glcm_grid)
        # lbp_hist.append(extractor.lbp_hist)
        add = corrected_address+ 'testing_img/' + classes_name[i] + '/image(' + str(j) + ').JPG'
        extractor.pre_process2(add)
        extractor.extract_haralick()
        red_h.append(extractor.red_haralick)
        blue_h.append(extractor.blue_haralick)    
        green_h.append(extractor.green_haralick)
        haralick.append(extractor.h_features)
        # plt.imsave(corrected_address + 'testing_img/' + classes_name[i] + '/image(' + str(j) + ').JPG', extractor.image_rgb)
        # plt.imsave(original_address + 'testing_img/' + classes_name[i] + '/image(' + str(j) + ').JPG', extractor.origin_rgb)


    adr = 'D:/Thesis_data/mlp_data/test_'
    # np.savez_compressed(adr+ 'overall_geometry_' + classes_name[i],  overall_geometry)
    # np.savez_compressed(adr+ 'overall_rgb_' + classes_name[i],  overall_rgb)
    # np.savez_compressed(adr+ 'overall_hsv_' + classes_name[i],  overall_hsv)
    # # np.savez_compressed(adr+ 'n1_' + classes_name[i],  n1)
    # # np.savez_compressed(adr+ 'structure_' + classes_name[i],  structure)
    # # np.savez_compressed(adr+ 'n2_' + classes_name[i],  n2)
    # # np.savez_compressed(adr+ 'moldered_' + classes_name[i],  moldered)
    # np.savez_compressed(adr+ 'color_grid_' + classes_name[i],  color_grid)
    # np.savez_compressed(adr+ 'glcm_grid_' + classes_name[i],  glcm_grid)
    np.savez_compressed(adr+ 'haralick_' + classes_name[i],  haralick)
    # np.savez_compressed(adr+ 'lbp_hist_' + classes_name[i],  lbp_hist)
    # np.savez_compressed(adr+ 'comp_hsv_' + classes_name[i],  comp_hsv)
    np.savez_compressed(adr+ 'red_haralick_' + classes_name[i],  red_h)
    np.savez_compressed(adr+ 'blue_haralick_' + classes_name[i],  blue_h)
    np.savez_compressed(adr+ 'green_haralick_' + classes_name[i],  green_h)



    overall_geometry = []
    overall_rgb = []
    overall_hsv = []
    n1 = []
    structure = []
    n2 = []
    moldered =[]
    glcm = []
    color_grid = []
    glcm_grid = []
    lbp_hist = []
    haralick = []
    comp_hsv = []
    red_h = []
    blue_h= []
    green_h = []

    for j in range(1,511):
        print(classes_name[i],'IMAGE ', j)
        # add = training_address + classes_name[i]+ '/image(' + str(j) + ').JPG'
        # image = cv2.imread(add)
        # extractor.extract(image)
        # overall_geometry.append(extractor.overall_geometry)
        # overall_rgb.append(extractor.overall_rgb_stat)
        # overall_hsv.append(extractor.overall_hsv_stat)
        # # n1.append(extractor.n1)
        # # structure.append(extractor.structure)
        # # n2.append(extractor.n2)
        # # moldered.append(extractor.mold)
        # color_grid.append(extractor.grid_stat)
        # glcm_grid.append(extractor.glcm_grid)
        # lbp_hist.append(extractor.lbp_hist)
        add = corrected_address+ 'training_img/' + classes_name[i] + '/image(' + str(j) + ').JPG'
        extractor.pre_process2(add)
        extractor.extract_haralick()
        haralick.append(extractor.h_features)
        red_h.append(extractor.red_haralick)
        blue_h.append(extractor.blue_haralick)
        green_h.append(extractor.green_haralick)
        # plt.imsave(corrected_address+ 'training_img/' + classes_name[i] + '/image(' + str(j) + ').JPG', extractor.image_rgb)
        # plt.imsave(original_address + 'training_img/' + classes_name[i] + '/image(' + str(j) + ').JPG', extractor.origin_rgb)

    adr = 'D:/Thesis_data/mlp_data/train_'
    # np.savez_compressed(adr+ 'overall_geometry_' + classes_name[i],  overall_geometry)
    # np.savez_compressed(adr+ 'overall_rgb_' + classes_name[i],  overall_rgb)
    # np.savez_compressed(adr+ 'overall_hsv_' + classes_name[i],  overall_hsv)
    # np.savez_compressed(adr+ 'n1_' + classes_name[i],  n1)
    # np.savez_compressed(adr+ 'structure_' + classes_name[i],  structure)
    # np.savez_compressed(adr+ 'n2_' + classes_name[i],  n2)
    # np.savez_compressed(adr+ 'moldered_' + classes_name[i],  moldered)
    # np.savez_compressed(adr+ 'color_grid_' + classes_name[i],  color_grid)
    # np.savez_compressed(adr+ 'glcm_grid_' + classes_name[i],  glcm_grid)
    np.savez_compressed(adr+ 'haralick_' + classes_name[i],  haralick)
    # np.savez_compressed(adr+ 'lbp_hist_' + classes_name[i],  lbp_hist)
    # np.savez_compressed(adr+ 'comp_hsv_' + classes_name[i],  comp_hsv)
    np.savez_compressed(adr+ 'red_haralick_' + classes_name[i],  red_h)
    np.savez_compressed(adr+ 'blue_haralick_' + classes_name[i],  blue_h)
    np.savez_compressed(adr+ 'green_haralick_' + classes_name[i],  green_h)


# def batch_join_file():


if __name__ == '__main__':
    k=0
    p1 = multiprocessing.Process(target=batch_prepare, args=(k+0,))
    p2 = multiprocessing.Process(target=batch_prepare, args=(k+1,))
    p3 = multiprocessing.Process(target=batch_prepare, args=(k+2,))
    p4 = multiprocessing.Process(target=batch_prepare, args=(k+3,))
    p5 = multiprocessing.Process(target=batch_prepare, args=(k+4,))
    p6 = multiprocessing.Process(target=batch_prepare, args=(k+5,))
    p7 = multiprocessing.Process(target=batch_prepare, args=(k+6,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()

    p1.join()
    p2.join()
    p3.join()  
    p4.join()
    p5.join()
    p6.join()
    p7.join()

    print("Cooling CPU!")
    for i in range(60):
        sys.stdout.write('-')
        sys.stdout.flush()
        time.sleep(0.2)

    k=7
    p1 = multiprocessing.Process(target=batch_prepare, args=(k+0,))
    p2 = multiprocessing.Process(target=batch_prepare, args=(k+1,))
    p3 = multiprocessing.Process(target=batch_prepare, args=(k+2,))
    p4 = multiprocessing.Process(target=batch_prepare, args=(k+3,))
    p5 = multiprocessing.Process(target=batch_prepare, args=(k+4,))
    p6 = multiprocessing.Process(target=batch_prepare, args=(k+5,))
    p7 = multiprocessing.Process(target=batch_prepare, args=(k+6,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()

    p1.join()
    p2.join()
    p3.join()  
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    
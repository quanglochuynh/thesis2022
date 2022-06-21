import multiprocessing
from matplotlib import image
import numpy as np
import pathlib
import cv2
import time
from image_extractor import feature_extract
import sys

# input_shape = (img_width, img_height, 3)
# train_dir = pathlib.Path('D:/Thesis_data/mlp_data/training_img')
# test_dir  = pathlib.Path('D:/Thesis_data/mlp_data/testing_img')
# checkpoint_dir = pathlib.Path('D:./TF_checkpoint/cacao_CNN/weight/')
# model_dir = pathlib.Path('D:./TF_backup/mlp/mlp.h5')
# model_plot_dir = pathlib.Path('D:./TF_backup/mlp/mlp.png')

classes_name = ['Agglutinated', 'Brittle', 'Compartmentalized_Brown', 'Compartmentalized_PartiallyPurple', 'Compartmentalized_Purple', 'Compartmentalized_Slaty', 'Compartmentalized_White', 'Flattened', 'Moldered', 'Plated_Brown', 'Plated_PartiallyPurple', 'Plated_Purple', 'Plated_Slaty', 'Plated_White']
training_address = 'D:/Thesis_data/mlp_data/training_img/'
testing_address = 'D:/Thesis_data/mlp_data/testing_img/'
# testing_address = 'D:/Thesis_data/Backups/Color_Corrected_512x512'

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
    
    # tmp = np.zeros((14))
    
    # t = tmp
    # t[i] = 1
    for j in range(1,121):
        print(classes_name[i],'IMAGE ', j)
        add = testing_address + classes_name[i]+ '/image(' + str(j) + ').JPG'
        image = cv2.imread(add)
        # extractor.extract(image)
        # overall_geometry = np.concatenate([overall_geometry, extractor.overall_geometry], axis=None)
        # overall_rgb = np.concatenate([overall_rgb, extractor.overall_rgb_stat], axis=None)
        # overall_hsv = np.concatenate([overall_hsv, extractor.overall_hsv_stat], axis=None)
        # n1 = np.concatenate([n1, extractor.n1], axis=None)
        # structure = np.concatenate([structure, extractor.structure], axis=None)
        # n2 = np.concatenate([n2, extractor.n2], axis=None)
        # moldered = np.concatenate([moldered, extractor.mold], axis=None)
        # color_grid = np.concatenate([color_grid, extractor.grid_stat], axis=None)
        extractor.pre_process(image)
        # extractor.extract_glcm()
        extractor.extract_lbp()
        # glcm.append(np.concatenate([extractor.glcm_asm, extractor.glcm_contrast, extractor.glcm_correlation, extractor.glcm_dissimilarity, extractor.glcm_energy, extractor.glcm_homogeneity], axis=None))
        lbp_hist.append(extractor.lbp_hist)

    adr = 'D:/Thesis_data/mlp_data/test_'
    # np.savez_compressed(adr+ 'overall_geometry_' + classes_name[i],  overall_geometry)
    # np.savez_compressed(adr+ 'overall_rgb_' + classes_name[i],  overall_rgb)
    # np.savez_compressed(adr+ 'overall_hsv_' + classes_name[i],  overall_hsv)
    # np.savez_compressed(adr+ 'n1_' + classes_name[i],  n1)
    # np.savez_compressed(adr+ 'structure_' + classes_name[i],  structure)
    # np.savez_compressed(adr+ 'n2_' + classes_name[i],  n2)
    # np.savez_compressed(adr+ 'moldered_' + classes_name[i],  moldered)
    # np.savez_compressed(adr+ 'color_grid_' + classes_name[i],  color_grid)
    # np.savez_compressed(adr+ 'glcm_grid_' + classes_name[i],  glcm_grid)
    # np.savez_compressed(adr+ 'glcm_2_' + classes_name[i],  glcm)
    np.savez_compressed(adr+ 'lbp_hist_' + classes_name[i],  lbp_hist)


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

    for j in range(1,481):
        print(classes_name[i],'IMAGE ', j)
        add = training_address + classes_name[i]+ '/image(' + str(j) + ').JPG'
        image = cv2.imread(add)
        # extractor.extract(image)
        # overall_geometry.append(extractor.overall_geometry)
        # overall_rgb.append(extractor.overall_rgb_stat)
        # overall_hsv.append(extractor.overall_hsv_stat)
        # n1.append(extractor.n1)
        # structure.append(extractor.structure)
        # n2.append(extractor.n2)
        # moldered.append(extractor.mold)
        # color_grid.append(extractor.grid_stat)
        extractor.pre_process(image)
        # extractor.extract_glcm()
        extractor.extract_lbp()
        # glcm.append(np.concatenate([extractor.glcm_asm, extractor.glcm_contrast, extractor.glcm_correlation, extractor.glcm_dissimilarity, extractor.glcm_energy, extractor.glcm_homogeneity], axis=None))
        lbp_hist.append(extractor.lbp_hist)
        

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
    # np.savez_compressed(adr+ 'glcm_2_' + classes_name[i],  glcm)
    np.savez_compressed(adr+ 'lbp_hist_' + classes_name[i],  lbp_hist)


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
        time.sleep(1)

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
    
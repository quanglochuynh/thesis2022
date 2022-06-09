import multiprocessing
from matplotlib import image
import numpy as np
import pathlib
import cv2
import time
from image_extractor import feature_extract

# input_shape = (img_width, img_height, 3)
# train_dir = pathlib.Path('D:/Thesis_data/mlp_data/training_img')
# test_dir  = pathlib.Path('D:/Thesis_data/mlp_data/testing_img')
# checkpoint_dir = pathlib.Path('D:./TF_checkpoint/cacao_CNN/weight/')
# model_dir = pathlib.Path('D:./TF_backup/mlp/mlp.h5')
# model_plot_dir = pathlib.Path('D:./TF_backup/mlp/mlp.png')

classes_name = ['Agglutinated', 'Brittle', 'Compartmentalized_Brown', 'Compartmentalized_PartiallyPurple', 'Compartmentalized_Purple', 'Compartmentalized_Slaty', 'Compartmentalized_White', 'Flattened', 'Moldered', 'Plated_Brown', 'Plated_PartiallyPurple', 'Plated_Purple', 'Plated_Slaty', 'Plated_White']
training_address = 'D:/Thesis_data/mlp_data/training_img/'
testing_address = 'D:/Thesis_data/mlp_data/testing_img/'

def batch_prepare(i):
    extractor = feature_extract()
    overall_geometry = np.array([], dtype=float)
    overall_rgb = np.array([], dtype=float)
    overall_hsv = np.array([], dtype=float)
    n1 = np.array([], dtype=float)
    structure = np.array([], dtype=float)
    n2 = np.array([], dtype=float)
    moldered = np.array([], dtype=float)
    color_grid = np.array([], dtype=float)
    glcm_grid = np.array([], dtype=float)
    
    # tmp = np.zeros((14))
    
    # t = tmp
    # t[i] = 1
    for j in range(1,121):
        print(classes_name[i],'IMAGE ', j)
        add = testing_address + classes_name[i]+ '/image(' + str(j) + ').JPG'
        image = cv2.imread(add)
        extractor.extract(image)
        overall_geometry = np.concatenate([overall_geometry, extractor.overall_geometry], axis=None)
        overall_rgb = np.concatenate([overall_rgb, extractor.overall_rgb_stat], axis=None)
        overall_hsv = np.concatenate([overall_hsv, extractor.overall_hsv_stat], axis=None)
        n1 = np.concatenate([n1, extractor.n1], axis=None)
        structure = np.concatenate([structure, extractor.structure])
        n2 = np.concatenate([n2, extractor.n2], axis=None)
        moldered = np.concatenate([moldered, extractor.mold], axis=None)
        color_grid = np.concatenate([color_grid, extractor.grid_stat], axis=None)
        glcm_grid = np.concatenate([glcm_grid, extractor.glcm_grid], axis=None)

    np.savez_compressed('D:/Thesis_data/mlp_data/Extracted_features_TEST_' + classes_name[i],  overall_geometry=overall_geometry, overall_rgb=overall_rgb, overall_hsv=overall_hsv, n1=n1, structure=structure, n2=n2, moldered=moldered, color_grid=color_grid, glcm_grid=glcm_grid)


    for j in range(1,481):
        print(classes_name[i],'IMAGE ', j)
        add = training_address + classes_name[i]+ '/image(' + str(j) + ').JPG'
        image = cv2.imread(add)
        extractor.extract(image)
        overall_geometry = np.concatenate([overall_geometry, extractor.overall_geometry], axis=None)
        overall_rgb = np.concatenate([overall_rgb, extractor.overall_rgb_stat], axis=None)
        overall_hsv = np.concatenate([overall_hsv, extractor.overall_hsv_stat], axis=None)
        n1 = np.concatenate([n1, extractor.n1], axis=None)
        structure = np.concatenate([structure, extractor.structure])
        n2 = np.concatenate([n2, extractor.n2], axis=None)
        moldered = np.concatenate([moldered, extractor.mold], axis=None)
        color_grid = np.concatenate([color_grid, extractor.grid_stat], axis=None)
        glcm_grid = np.concatenate([glcm_grid, extractor.glcm_grid], axis=None)

    np.savez_compressed('D:/Thesis_data/mlp_data/Extracted_features_TRAIN_' + classes_name[i], overall_geometry=overall_geometry, overall_rgb=overall_rgb, overall_hsv=overall_hsv, n1=n1, structure=structure, n2=n2, moldered=moldered, color_grid=color_grid, glcm_grid=glcm_grid)

# print(x_test)
# print(y_test)

# data = np.load('D:/Thesis_data/mlp_data/XY_train.npz', allow_pickle=True)
# a = data['xtest']
# print(a)
# b = data['ytest']
# print(b)

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

    print("Done!")
    time.sleep(60)

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
    
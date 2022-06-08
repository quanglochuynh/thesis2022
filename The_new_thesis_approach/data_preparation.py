import multiprocessing
from matplotlib import image
import numpy as np
import pathlib
import cv2
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
    x_train = np.array([], dtype=float)
    y_train = np.array([], dtype=float)
    x_test = np.array([], dtype=float)
    y_test = np.array([], dtype=float)
    tmp = np.zeros((14))
    
    t = tmp
    t[i] = 1
    for j in range(1,121):
        print(classes_name[i],'IMAGE ', j)
        add = testing_address + classes_name[i]+ '/image(' + str(j) + ').JPG'
        image = cv2.imread(add)
        extractor.extract(image)
        res = np.concatenate([extractor.overall_rgb_stat, extractor.overall_hsv_stat, extractor.overall_geometry, extractor.n1, extractor.structure, extractor.n2, extractor.mold], axis=None)
        x_test = np.append(x_test, res)
        y_test = np.append(y_test, t)
        # print(np.shape(res))
        # print(np.shape(extractor.n1))
        # break
    np.savez_compressed('D:/Thesis_data/mlp_data/XY_test_3_' + classes_name[i], xtest = x_test, ytest = y_test)

    t = tmp
    t[i] = 1
    for j in range(1,481):
        print(classes_name[i],'IMAGE ', j)
        add = training_address + classes_name[i]+ '/image(' + str(j) + ').JPG'
        image = cv2.imread(add)
        extractor.extract(image)
        res = np.concatenate([extractor.overall_rgb_stat, extractor.overall_hsv_stat, extractor.overall_geometry, extractor.n1, extractor.structure, extractor.n2, extractor.mold], axis=None)
        x_train = np.append(x_train, res)
        y_train = np.append(y_train, t)
    np.savez_compressed('D:/Thesis_data/mlp_data/XY_train_3_' + classes_name[i], xtrain = x_train, ytrain = y_train)

# print(x_test)
# print(y_test)

# data = np.load('D:/Thesis_data/mlp_data/XY_train.npz', allow_pickle=True)
# a = data['xtest']
# print(a)
# b = data['ytest']
# print(b)

if __name__ == '__main__':
    k=8
    p1 = multiprocessing.Process(target=batch_prepare, args=(k+0,))
    p2 = multiprocessing.Process(target=batch_prepare, args=(k+1,))
    p3 = multiprocessing.Process(target=batch_prepare, args=(k+2,))
    p4 = multiprocessing.Process(target=batch_prepare, args=(k+3,))
    p5 = multiprocessing.Process(target=batch_prepare, args=(k+4,))
    p6 = multiprocessing.Process(target=batch_prepare, args=(k+5,))
    p7 = multiprocessing.Process(target=batch_prepare, args=(0,))

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

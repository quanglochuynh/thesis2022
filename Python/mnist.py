import numpy as np
import gzip as gz
import cv2

dir = "D:/Thesis_data/MNIST/"
train_img_dir = "train-images-idx3-ubyte.gz"
train_lab_dir = "train-labels-idx1-ubyte.gz"
test_img_dir  = "t10k-images-idx3-ubyte.gz"
test_lab_dir  = "t10k-labels-idx1-ubyte.gz"

inp_training = gz.open(dir+train_img_dir, 'r')

im_size = 28

num_images = 5

inp_training.read(16)

buf = inp_training.read(im_size * im_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, im_size, im_size, 1)

image = np.asarray(data[1]).squeeze()
image = cv2.resize(image, (280,280), interpolation=cv2.INTER_NEAREST)
cv2.imshow("image",image)

cv2.waitKey(0)
cv2.destroyAllWindows()
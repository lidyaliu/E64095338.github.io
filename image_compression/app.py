%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time

from PIL import Image

img = Image.open('input/test1 (1).png')
imggray = img.convert('LA')
plt.figure(figsize=(9, 6))
plt.imshow(imggray);

#convert the image data into a numpy matrix, plotting the result to show the data is unchanged
imgmat = np.array(list(imggray.getdata(band=0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)
plt.figure(figsize=(9,6))
plt.imshow(imgmat, cmap='gray');

U, sigma, V = np.linalg.svd(imgmat) #to compute the  singular value decomposition

#computing approx. of the img using first column of U and first row of V reproduces the most prominent feature of the image
reconstimg = np.matrix(U[:, :1]) * np.diag(sigma[:1]) * np.matrix(V[:1, :])
plt.imshow(reconstimg, cmap='gray');

#the loop shows the reconstructed image using the first n vectors of the SVD(n is shown on title of plot)
#The first 50 vectors produce an image very close the original image, while taking up only  (50∗3900+50+50∗2600)/(3900∗2600)≈3.2% as much space as the original data
for i in range(2, 51, 5):
    reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
    plt.imshow(reconstimg, cmap='gray')
    title = "n = %s" % i
    plt.title(title)
    plt.show()


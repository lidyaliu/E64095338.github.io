# Image-Compression Tool using SVD with Mercury

112-2 國立成功大學編譯系統作業 1 初始檔案

## Environmental Setup
- Install Dependencies

```shell
pip install mercury
```

or

```shell
conda install -c conda-forge mercury
```

## About Singular Value Decomposition

A matrix with dimensions of m × n is a real number grid with m rows and n columns. Matrix sizes of m×n explain linear mappings from n-dimensional to m-dimensional space in the mathematical field of linear algebra. In general, the definition of linear is that the origin in n-dimensional space maps to the origin in m-dimensional space, and straight lines map to straight lines. We may compute the product AB, which is a (m × k)‑matrix, when we have a (m × n)‑matrix A and a (n × k)‑matrix B. The mapping that corresponds to AB is precisely made up of the mappings that correspond to A and B, in that order.

image

From **Singular Value Decomposition (SVD)**, each (m×n)‑matrix A can be expressed as a product of U and V, which are orthogonal matrices, and Σ, the matrix, which is made up of zeros everywhere else and descending non-negative values on its diagonal. The singular values (SVs) of A are the entries σ1 ≥ σ2 ≥ σ3 ≥ … ≥ 0 on the diagonal of Σ. In a geometric sense, Σ scales by σj to transfer the j-th coordinate vector of n-dimensional space to the j-th coordinate vector of m-dimensional space. When U and V are orthogonal, it suggests that they correspond to rotations of m-dimensional and n-dimensional space, respectively, potentially followed by a reflection. As a result, the only vector length that is altered by Σ.

## Using SVD for image compression

A given image can be divided into its three color channels, red, green, and blue. Any value between 0 and 255 can be represented by a (m×n)‑matrix for each channel. The matrix A, which represents one of the channels, will now be compressed.

In order to accomplish this, we compute an approximation to the matrix A that requires a very small amount of storage space. The best part of SVD is that it sorts the data in the matrices U, Ϋ, and V according to how big of a contribution they make to matrix A in the product. This allows us to obtain a fairly accurate approximation by utilizing only the most significant components of the matrices.

Now, we select a number k of singular values to be used in the approximation. The quality of the approximation improves with a greater number, but encoding it requires more data. Now, we only consider the upper left (k × k)-square of Σ, which contains the k greatest (and thus most significant) singular values, and the first k columns of U and V. Having: 

image

The colored region indicates how much data is required to store this approximation:
compressed size = k × (1 + m + n) = m×k + k + k˗ n
(In actuality, since U and V are orthogonal, significantly less space is required.) It is possible to demonstrate that this approximation is, in some ways, ideal.
Principal component analysis in statistics and model reduction through numerical simulations both frequently use SVD. More advanced techniques, such as JPG, which consider human perception, typically do better than SVD compression when it comes to image compression.

## About This Project

I used Anaconda to construct the .ipynb file and utilized Mercury for the UI/UX and functionally used as Web Application. 
Provided the python code below:

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import mercury as mr
file = mr.File(label="File upload", max_file_size="10MB")
print(f"Uploaded Image: {file.filepath}\{file.filename}")
img = Image.open(file.filepath) #Load the image
imggray = img.convert('LA') #Convert image to grayscale for comparison convenience
plt.figure(figsize=(9, 6))
plt.imshow(imggray) #Display original grayscale image
plt.title('Original Image (Grayscale)')
plt.axis('off')
plt.show()
imgmat = np.array(list(imggray.getdata(band=0)), float) #Convert image data into a numpy matrix
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)
plt.figure(figsize=(9, 6))
plt.imshow(imgmat, cmap='gray') #Display grayscale image matrix
plt.title('Image Matrix')
plt.axis('off')
plt.show()
U, sigma, V = np.linalg.svd(imgmat) #Perform Singular Value Decomposition (SVD)
#Reconstruct the image using different numbers of singular vectors
for i in range(1, min(imgmat.shape) + 1,min(imgmat.shape)//10): #min(imgmat.shape) represent max number of singular vectors possible
    new = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
    plt.figure(figsize=(9, 6))
    plt.imshow(new, cmap='gray')
    plt.title("n = %s" % i)
    plt.axis('off')
    plt.show()
```

using Windows PowerShell, run mercury on the same directory as the .ipynb file, using the syntax below. (Provided mercury should be installed before)

```shell
mercury run
```





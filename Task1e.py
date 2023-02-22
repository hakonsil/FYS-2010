import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#importing the restored images
jupiter1 = cv2.imread(r'Plots\restored_jupiter1.png')
jupiter2 = cv2.imread(r'Plots\restored_jupiter2.png')

"""STOLEN"""
# Method to process the red band of the image
bing = 20
bong = 240
def normalizeRed(intensity):
    iI      = intensity
    minI    = bing
    maxI    = bong
    minO    = 0
    maxO    = 255
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

def normalizeGreen(intensity):
    iI      = intensity
    minI    = bing
    maxI    = bong
    minO    = 0
    maxO    = 255
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

def normalizeBlue(intensity):
    iI      = intensity
    minI    = bing
    maxI    = bong
    minO    = 0
    maxO    = 255
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

"""imageObject = Image.open(r'Plots\restored_jupiter1.png')
Bands = imageObject.split()
redBand = Bands[0].point(normalizeRed)
greenBand = Bands[1].point(normalizeGreen)
blueBand = Bands[2].point(normalizeBlue)
normalized = Image.merge("RGB", (redBand, greenBand, blueBand))
normalized.show()
normalized = np.array(normalized, dtype = 'uint8')
"""
for gamma in [1.15, 1.175, 1.2, 1.225, 1.25]:
    # Apply gamma correction.
    gamma_corrected = np.array(255*(jupiter1 / 255) ** gamma, dtype = 'uint8')
    # Save edited images.
    #cv2.imshow('gamma_transformed'+str(gamma)+'.jpg', gamma_corrected)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def laplacian(M, N):
    filter = np.zeros((M, N, 2), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            filter[i][j] = -((i-M/2)**2 + (j-N/2)**2)
    return filter

jupiter1 = cv2.cvtColor(jupiter1, cv2.COLOR_BGR2GRAY)
filter = laplacian(jupiter1.shape[0], jupiter1.shape[1])

ft = cv2.dft(np.float32(jupiter1), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(ft)
filtered = dft_shift * filter
f_ishift = np.fft.ifftshift(filtered)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
filterMag = 20 * np.log(cv2.magnitude(filter[:, :, 0], filter[:, :, 1]))
imgplot = plt.imshow(img_back, cmap="gray")
filterMag = np.array(filterMag, dtype = 'uint8')
plt.show()
cv2.imshow("lap_output.png", filterMag)
cv2.waitKey(0)
cv2.destroyAllWindows()

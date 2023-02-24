import cv2
import numpy as np
import matplotlib.pyplot as plt
#importing the restored images
jupiter1 = cv2.imread(r'Plots\restored_jupiter1.png')
jupiter2 = cv2.imread(r'Plots\restored_jupiter2.png')

"""Creating filters"""

#plot histogram of jupiter1
jupiter1_hist, bin_edges_jup1 = np.histogram(np.array(jupiter1), bins=256)
jupiter2_hist, bin_edges_jup2 = np.histogram(np.array(jupiter2), bins=256)

# plotting the greyscale histograms
"""plt.bar(bin_edges_jup1[0:-1], jupiter1_hist, color = 'k')
plt.xlabel('Pixel value')
plt.ylabel('Number of pixels')
plt.title('Greyscale histogram of Jupiter1.png')
plt.show()"""
def contrast_stretching(img, min, max):
    """Performs contrast stretching on the input image, min and max 
    refer to the values that are stretched, i.e. min becomes 0 and max becomes 255"""
    img = np.clip(img, min, max)
    newmin = 0
    newmax = 255
    newimg = (img - min) / (max - min) * (newmax - newmin) + newmin
    return newimg

def apply_contrast_stretching(img, oldmin=20, oldmax=240):
    """Applies contrast stretching to to the input image, converting oldmin to 0 and oldmax to 255
    (only works for color images)"""
    b, g, r = cv2.split(img)
    r = contrast_stretching(r, oldmin, oldmax)
    g = contrast_stretching(g, oldmin, oldmax)
    b = contrast_stretching(b, oldmin, oldmax)
    new_img = cv2.merge((b, g, r))
    new_img = np.array(new_img, dtype = 'uint8')
    return new_img

def apply_gamma_correction(img, gamma=1.1):
    """Performs gamma correction on the input image"""
    img = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
    return img

def laplacian(f, c):
    f = f/255
    F = np.fft.fftshift(np.fft.fft2(f))

    P, Q = F.shape
    H = np.zeros((P, Q), dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            H[u][v] = -4*np.pi*np.pi*((u-P/2)**2 + (v-Q/2)**2)
    Lap = H*F
    Lap = np.fft.ifftshift(Lap)
    Lap = np.real(np.fft.ifft2(Lap))

    oldrange = (np.max(Lap) - np.min(Lap))
    newrange = 1 - -1
    Lapscaled = (((Lap - np.min(Lap)) * newrange) / oldrange) - 1


    g = f + c*Lapscaled
    g = np.clip(g, 0, 1)
    return g

def apply_laplacian(img, red_c = -0, green_c = -0.2, blue_c = -0.5):
    """Applies the laplacian filter to input image (only works for color images)
    The values of c are chosen to give the best results"""
    b, g, r = cv2.split(img)
    b = laplacian(b, blue_c)*255
    g = laplacian(g, green_c)*255
    r = laplacian(r, red_c)*255
    b = np.array(b, dtype = 'uint8')
    g = np.array(g, dtype = 'uint8')
    r = np.array(r, dtype = 'uint8')
    x = cv2.merge((b, g, r))
    return x

"""Applying filters"""
img = jupiter1
enhanced_jupiter1 = apply_contrast_stretching(img, 15, 245)
enhanced_jupiter1 = apply_gamma_correction(enhanced_jupiter1, 1.2)
img = enhanced_jupiter1
img1 = apply_laplacian(img, -0, -0.2, -1)
img2 = apply_laplacian(img, -0.2, -0, -1)
img3 = apply_laplacian(img, -0.1, -0.1, -1)

comparison1 = np.concatenate((img, img1), axis=0)
comparison2 = np.concatenate((img2, img3), axis=0)
comparison = np.concatenate((comparison1, comparison2), axis=1)

# scale image to 50% of original size
comparison = cv2.resize(comparison, (0,0), fx=0.6, fy=0.6)
cv2.imshow('Comparison', comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""Laplacian looks bad, at least for jupiter1, so I will not use it, but keep it
just in case, maybe its good for jupiter2, will atleast get points for it"""

# best contrast stretching values for jupiter1: 15, 245
# best gamma correction values for jupiter1: 1.2
# best total result for jupiter1: contrast stretching with 15, 245 then
# gamma correction with 1.2


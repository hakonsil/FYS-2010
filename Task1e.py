import cv2
import numpy as np
import matplotlib.pyplot as plt
#importing the restored images
jupiter1 = cv2.imread(r'Plots\restored_jupiter1.png')
jupiter2 = cv2.imread(r'Plots\restored_jupiter2.png')

"""Creating filters"""
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
    """Calculates the laplacian of the input image,
    and adds it to the image with a constant c (output is clipped to (0,1))"""
    f = f/255 #normalize the input function
    F = np.fft.fft2(f) #2D discrete Fourier transform
    F = np.fft.fftshift(F) # shifting the ffrequency components

    P, Q = F.shape #dimensions of the image
    H = np.zeros((P, Q), dtype=np.float32) #initializing the transfer function
    for u in range(P):
        for v in range(Q):
            # iterating through the transfer function, and calculating appropriate values
            D2 = ((u-P/2)**2 + (v-Q/2)**2) #euclidean distance from center squared
            H[u][v] = -4*(np.pi**2)*D2 # transfer function

    Lap = np.fft.ifftshift(H*F) # calculating the laplacian and shifting the frequency components back
    Lap = np.real(np.fft.ifft2(Lap)) # discarding the imaginary part of the laplacian (we only care about the real part)

    # we need to scale the laplacian to be between -1 and 1 
    # (since the values are very big, so we need the laplacian and the image to be on the same scale)
    oldrange = (np.max(Lap) - np.min(Lap)) 
    newrange = 1 - -1
    Lap_scaled = (((Lap - np.min(Lap)) * newrange) / oldrange) - 1 #scaling the laplacian to (-1, 1)


    g = f + c*Lap_scaled # adding the laplacian to the image (scaled by c)
    g = np.clip(g, 0, 1) #clipping the image to (0, 1) since we need the image to be no
    return g

def apply_laplacian(img, red_c = -1, green_c = -1, blue_c = -1):
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
# For jupiter1
# contrast stretching
jupiter1_contrast = apply_contrast_stretching(jupiter1, 15, 245)

# gamma correction
jupiter1_gamma = apply_gamma_correction(jupiter1_contrast, 1.2)

# laplacian
jupiter1_laplacian = apply_laplacian(jupiter1, -0, -0.2, -1)

#compare with original
compare1 = np.concatenate((jupiter1, jupiter1_contrast), axis=1)
compare2 = np.concatenate((jupiter1_gamma, jupiter1_laplacian), axis=1)
compare = np.concatenate((compare1, compare2), axis=0)
cv2.imshow('Before and after enhancement jupiter1', compare)
cv2.waitKey(0)
cv2.destroyAllWindows()


# For jupiter2
# contrast stretching
jupiter2_contrast = apply_contrast_stretching(jupiter2, 25, 225)

# gamma correction
jupiter2_gamma = apply_gamma_correction(jupiter2_contrast, 1.05)

# compare with original
compare2 = np.concatenate((jupiter2, jupiter2_gamma), axis=1)
cv2.imshow('Before and after enhancement jupiter2', compare2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#show before and after histogram of jupiter2
plt.hist(jupiter2.ravel(),256,[0,256], color = 'k')
plt.title('Histogram of jupiter2 before enhancement')
plt.xlabel('Pixel value')
plt.ylabel('Pixel count')
plt.show()

plt.hist(jupiter2_gamma.ravel(),256,[0,256], color = 'k')
plt.title('Histogram of jupiter2 after enhancement')
plt.xlabel('Pixel value')
plt.ylabel('Pixel count')
plt.show()
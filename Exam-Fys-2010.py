from PIL import Image
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt


# test test test 


"""TASK 1"""
# importing the images
jupiter1 = cv2.imread(r'Homeexam supplementary data\Jupiter1.png')
jupiter2 = cv2.imread(r'Homeexam supplementary data\Jupiter2.png')

"""Task b"""
# creating the histograms
jupiter1_hist, bin_edges_jup1 = np.histogram(np.array(jupiter1), bins=256)
jupiter2_hist, bin_edges_jup2 = np.histogram(np.array(jupiter2), bins=256)

# plotting the greyscale histograms
plt.bar(bin_edges_jup2[0:-1], jupiter2_hist, color = 'k')
plt.xlabel('Pixel value')
plt.ylabel('Number of pixels')
plt.title('Greyscale histogram of Jupiter2.png')
#plt.show()

plt.bar(bin_edges_jup1[0:-1], jupiter1_hist, color = 'k')
plt.xlabel('Pixel value')
plt.ylabel('Number of pixels')
plt.title('Greyscale histogram of Jupiter1.png')
#plt.show()

# creating the color histograms
"""STOLEN CODE REWRITE BEFORE SUBMITTING!!! (only the color histogram part))"""
colors = ("red", "green", "blue")
for channel_id, color in enumerate(colors):
    histogram, bin_edges = np.histogram(
        np.array(jupiter1)[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=color)

plt.title("Color Histogram of Jupiter1.png")
plt.xlabel("Color value")
plt.ylabel("Pixel count")
#plt.show()

"""Tasks c & d"""
# applying median filter to remove salt and pepper noise
median_jupiter2 = cv2.medianBlur(jupiter2, 3)
"""cv2.imshow('Median filtered image', median_jupiter2)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    """PLAGIARIZED CODE REWRITE BEFORE SUBMITTING!!!
    https://stackoverflow.com/questions/65483030/notch-reject-filtering-in-python"""
    P, Q = shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1.0

    return H

def filter_img(img):
    """REWRITE BEFORE SUBMITTING!!!"""
    blue,green,red = cv2.split(img)
    H = notch_reject_filter(blue.shape, 0.9, 5, 0)
    blue = np.fft.fft2(blue)
    blue = np.fft.fftshift(blue)
    blue = blue*H
    blue = np.fft.ifftshift(blue)
    blue = np.fft.ifft2(blue)
    blue = abs(blue)

    green = np.fft.fft2(green)
    green = np.fft.fftshift(green)
    green = green*H
    green = np.fft.ifftshift(green)
    green = np.fft.ifft2(green)
    green = abs(green)

    red = np.fft.fft2(red)
    red = np.fft.fftshift(red)
    red = red*H
    red = np.fft.ifftshift(red)
    red = np.fft.ifft2(red)
    red = abs(red)

    img = cv2.merge((blue,green,red))
    img = np.array(img, dtype=np.uint8)

    return img


#plot magnitude spectrum of original image
plt.clf()
a = np.fft.fft2(cv2.cvtColor(median_jupiter2, cv2.COLOR_BGR2GRAY))
a = np.fft.fftshift(a)
a = np.log(np.abs(a))
plt.imshow(a, cmap='gray')
plt.title('Magnitude spectrum of median filtered image')
plt.show()

#plot magnitude spectrum of filter
b = notch_reject_filter(a.shape, 1, 5, 0)*notch_reject_filter(a.shape, 0.9, 7, 0)
plt.imshow(b*a, cmap='gray')
plt.title('Magnitude spectrum of notch reject filter')
plt.show()


filtered_jupiter2 = filter_img(median_jupiter2)
cv2.imshow('filtered image', filtered_jupiter2) 
cv2.waitKey(0) # keeps image open until any key is pressed
cv2.destroyAllWindows()
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""TASK 1"""
# importing the images
jupiter1 = cv2.imread(r'Homeexam supplementary data\Jupiter1.png')
jupiter2 = cv2.imread(r'Homeexam supplementary data\Jupiter2.png')

"""Task b"""
# creating the histograms
jupiter1_hist, bin_edges_jup1 = np.histogram(np.array(jupiter1), bins=256)
jupiter2_hist, bin_edges_jup2 = np.histogram(np.array(jupiter2), bins=256)

# plotting the greyscale histograms
plt.bar(bin_edges_jup1[0:-1], jupiter1_hist, color = 'k')
plt.xlabel('Pixel value')
plt.ylabel('Number of pixels')
plt.title('Greyscale histogram of Jupiter1.png')
plt.show()

plt.bar(bin_edges_jup2[0:-1], jupiter2_hist, color = 'k')
plt.xlabel('Pixel value')
plt.ylabel('Number of pixels')
plt.title('Greyscale histogram of Jupiter2.png')
plt.show()

# creating the color histogram of jupiter1
for i in range(0, 3):
    histogram, bin_edges = np.histogram(np.array(jupiter1)[:, :, i], bins=256, range=(0, 256))
    plt.plot(bin_edges[0:-1], histogram, color='bgr'[i])
plt.title("Color Histogram of Jupiter1.png")
plt.xlabel("Color value")
plt.ylabel("Pixel count")
plt.show()

"""Tasks c & d"""
def notch_reject_filter(shape, Q, u_k, v_k):
    """Creates a notch reject filter with a circle of zeros around (u_k, v_k) with radius Q"""
    M, N = shape
    # Initialize filter with zeros
    H = np.zeros((M, N))

    # Traverse through filter
    for u in range(0, M):
        for v in range(0, N):
            #find the distance from the center (using pythagoras)
            D_k = np.sqrt((u - M/2 - u_k)**2 + (v - N/2 - v_k)**2)
            D_min_k = np.sqrt((u - M/2 + u_k)**2 + (v - N/2 + v_k)**2)

            if D_min_k <= Q or D_k <= Q:
                H[u, v] = 0.0 # creating a circle of zeros around with radius Q at pos (u_k, v_k)
            else:
                H[u, v] = 1.0 # set to one everywhere else
    return H

# applying the filters to jupiter1

# Firstly I split the image into its separate color channels
# since the different color channels are not affected by the same noise
# so they need different filters
blue, green, red = cv2.split(jupiter1) # separating color channels

# creating the magnitude spectrum of the red channel
# so we can see which frequencies to filter out
magnitude_img = np.fft.fft2(red)
magnitude_img = np.fft.fftshift(magnitude_img)
magnitude_img = np.log(np.abs(magnitude_img))

# creating the notch reject filter
H1 = notch_reject_filter(red.shape, Q= 3.0, u_k=7, v_k=7) #blocking out the bright spots
H2 = np.full((red.shape), 1.0)
for i in range (red.shape[0]): #blocking out the bright lines
    for j in range (red.shape[0]):
        if i == 249 and j <249:
            H2[i, j] = 0.0
        elif j == 249 and i < 249:
            H2[i, j] = 0.0
        elif i == 263 and j > 263:
            H2[i,j] = 0.0
        elif j == 263 and i > 263:
            H2[i,j] =0.0
        elif i == 249 and j > 263:
            H2[i, j] = 0.0
        elif j == 249 and i > 263:
            H2[i, j] = 0.0
        elif i == 263 and j < 249:
            H2[i, j] = 0.0
        elif j == 263 and i < 249:
            H2[i, j] = 0.0
H = H1*H2

# applying the nr filter to the red channel
red = np.fft.fft2(red) # fourier transform
red = np.fft.fftshift(red) # shift zero frequency to center
red = red*H # apply filter
red = np.fft.ifftshift(red) # shift zero frequency back
red = np.fft.ifft2(red) # inverse fourier transform
red = np.abs(red) # get magnitude

# plotting the magnitude spectrum of the red channel and the filter

plt.imshow(H*magnitude_img, cmap='gray')
plt.title('Notch reject filter applied to red channel')
plt.axis('off')
plt.show()

plt.imshow(magnitude_img, cmap='gray')
plt.title('Magnitude spectrum of red channel')
plt.axis('off')
plt.show()
red = np.array(red, dtype=np.uint8)# converting back to uint8 (the 'correct' data type)

# then we will apply a median filter (twice) to the blue channel
blue = cv2.medianBlur(cv2.medianBlur(blue, 3), 3)
blue = np.array(blue, dtype=np.uint8)

# then we will remove the pepper in the green channel
# by applying a contraharmonic mean filter (Q=0 so its actually just an arithmetic mean filter)
def contraharmonic_mean(img, size, Q):
    kernel = np.full(size, 1.0) # creating the kernel
    a = cv2.filter2D((np.power(img, (Q + 1))), -1, kernel, borderType=cv2.BORDER_REPLICATE)
    b = cv2.filter2D((np.power(img, Q)), -1, kernel,  borderType=cv2.BORDER_REPLICATE)
    filtered = a/b
    return filtered

green = contraharmonic_mean(green,(3,3), 0.0)
green = np.array(green, dtype=np.uint8)

# then we will merge the color channels back together
restored_jupiter1 = cv2.merge((blue, green, red))
restored_jupiter1 = np.array(restored_jupiter1, dtype=np.uint8)

#plotting the final result
cv2.imshow('Restored Jupiter1', restored_jupiter1)
cv2.waitKey(0)
cv2.destroyAllWindows()



# applying the filters to jupiter2
def filter_img(img, H):
    """Applies a filter H on an image img (filter is applied on all color channels)"""
    #plot magnitude spectrum of image
    magnitude_img = np.fft.fft2(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    magnitude_img = np.fft.fftshift(magnitude_img)
    magnitude_img = np.log(np.abs(magnitude_img))

    blue,green,red = cv2.split(img) # separating color channels
    color_channels = [blue, green, red]
    for i in range(3):
        color_channels[i] = np.fft.fft2(color_channels[i]) # fourier transform
        color_channels[i] = np.fft.fftshift(color_channels[i]) # shift zero frequency to center
        color_channels[i] = color_channels[i]*H # apply filter
        color_channels[i] = np.fft.ifftshift(color_channels[i]) # shift zero frequency back
        color_channels[i] = np.fft.ifft2(color_channels[i]) # inverse fourier transform
        color_channels[i] = np.abs(color_channels[i]) # get magnitude

    img = cv2.merge((color_channels[0], color_channels[1], color_channels[2]))
    img = np.array(img, dtype=np.uint8)

    #plotting the magnitude spectrum of the image and the filter
    plt.imshow(H*magnitude_img, cmap='gray')
    plt.title('Magnitude spectrum of notch reject filter')
    plt.axis('off')
    plt.show()
    plt.imshow(magnitude_img, cmap='gray')
    plt.title('Magnitude spectrum of image')
    plt.axis('off')
    plt.show()
    return img

# applying the filters to jupiter2
H = notch_reject_filter(jupiter2[:,:,0].shape, Q=1.0, u_k=5, v_k=0)
notchfiltered_jupiter2 = filter_img(jupiter2, H)
restored_jupiter2 = cv2.medianBlur(cv2.medianBlur(notchfiltered_jupiter2, 3), 3)

#plotting the final result
cv2.imshow('Restored Jupiter2', restored_jupiter2)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(0, 3):
    histogram, bin_edges = np.histogram(np.array(jupiter1)[:, :, i], bins=256, range=(0, 256))
    plt.plot(bin_edges[0:-1], histogram, color='bgr'[i])
plt.title("Color Histogram of Jupiter1.png")
plt.xlabel("Color value")
plt.ylabel("Pixel count")
plt.show()

for i in range(0, 3):
    histogram, bin_edges = np.histogram(np.array(restored_jupiter1)[:, :, i], bins=256, range=(0, 256))
    plt.plot(bin_edges[0:-1], histogram, color='bgr'[i])
plt.title("Color Histogram of restored Jupiter1.png")
plt.xlabel("Color value")
plt.ylabel("Pixel count")
plt.show()
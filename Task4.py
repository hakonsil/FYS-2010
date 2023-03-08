import numpy as np
import matplotlib.pyplot as plt

#importing the data files
point_cloud = np.load(r'Homeexam supplementary data\point_cloud.npz')
point_cloud_laplacian = np.load(r'Homeexam supplementary data\point_cloud_Laplacian.npz')

#extracting the data into arrays
X = point_cloud['X'] #coordinates
Z = point_cloud['Z'] #greyscale values
L = point_cloud_laplacian['L'] #Laplacian

"""---Task 4.1---"""
# creating a scatter plot of the data with (0,0) at top left corner
plt.scatter(X[:,1], X[:,0], c=Z, cmap='gray')
plt.ylim(400, 0)
plt.show()

"""---Task 4.2---"""
#plot histogram of Z
plt.hist(Z, bins=100, color='k')
plt.title('Histogram of z')
plt.xlabel('Pixel value')
plt.ylabel('Pixel count')
plt.show()

"""---Task 4.3---"""
# computing the eigenvalues and eigenvectors of the laplacian (frequencies and fourier modes)
eigenvalues, eigenvectors = np.linalg.eigh(L)

# plotting the eigenvalues
plt.plot(eigenvectors[:,2000], color='k')
plt.title('Eigenvalues')
plt.xlabel('Eigenvalue number')
plt.ylabel('Eigenvalue')
plt.show()

#implementing a few checks to make sure the eigenvalues are sorted and positive and that the eigenvectors are the right dimensions
# checking the dimensions of the eigenvectors
print(eigenvectors.shape)

# check if we have all positive values
if np.min(eigenvalues) < 0:
    print('The eigenvalues are not all positive...')

#check if the eigenvalues are sorted
for i in range(len(eigenvalues)-1):
    if eigenvalues[i] > eigenvalues[i+1]:
        print('The eigenvalues are not sorted...')
        break

"""---Task 4.4---"""
# plotting the fourier modes associated to the n-th eigenvalue
# note that the eigenvectors are the columns in the eigenvectors matrix, not the rows
n=10
plt.scatter(X[:,0], X[:,1], c=eigenvectors[:,n], cmap='gray')
plt.axis('off')
plt.ylim(400, 0)
plt.title('Eigenvalue number: 10')
plt.show()


"""---Task 4.5---"""
# computing the graph fourier transform of z
GFT = eigenvectors.T @ Z # computing the graph fourier transform

# plotting the graph fourier transform of z
plt.plot(eigenvalues, GFT, color='k')
plt.title('Graph fourier transform of z')
plt.xlabel('Eigenvalue (frequency))')
plt.ylabel('$\mathcal{GF}(z)$')
plt.show()

# double checking the dimensions of GFT
print(GFT.shape)


"""---Task 4.6---"""
# creating the low-pass filter with cutoff frequency c
c=3.5 # cutoff frequency
filter = np.copy(eigenvalues)
for i in range(len(filter)):
    if filter[i] < c:
        filter[i] = 1
    else:
        filter[i] = 0

lowpassfiltered_GFT = GFT*filter # filtering the gft

# plotting the filter
plt.plot(filter)
plt.title('Low pass filter')
plt.xlabel('Eigenvalue (frequency))')
plt.ylabel('Filter value')
plt.show()

# plotting the filtered graph fourier transform
plt.plot(eigenvalues,lowpassfiltered_GFT, color='k')
plt.title('Low pass filtered graph fourier transform of z, c='+ str(c))
plt.xlabel('Eigenvalue (frequency))')
plt.ylabel('$\mathcal{GF}(z)$')
plt.show()

# computing the inverse graph fourier transform of the filtered graph fourier transform of z
lowpassfiltered_Z = eigenvectors @ lowpassfiltered_GFT

# plotting the filtered data and the original data
plt.subplot(1,2,1)
plt.scatter(X[:,1], X[:,0], c=lowpassfiltered_Z, cmap='gray')
plt.axis('off')
plt.ylim(600, 0)
plt.title('Low pass filtered data, c='+ str(c))
plt.subplot(1,2,2)
plt.scatter(X[:,1], X[:,0], c=Z, cmap='gray')
plt.axis('off')
plt.ylim(600, 0)
plt.title('Original data')
plt.show()


"""---Task 4.7---"""
# creating the high-pass filter with cutoff frequency c
c=0.3 # cutoff frequency
filter = np.copy(eigenvalues)
for i in range(len(filter)):
    if filter[i] < c:
        filter[i] = 1
    else:
        filter[i] = 0

highpassfiltered_GFT = GFT*(1-filter) # filtering the gft

# computing the inverse graph fourier transform of the filtered graph fourier transform of z
highpassfiltered_Z = eigenvectors @ highpassfiltered_GFT

# plotting the filter
plt.plot(1-filter)
plt.title('High pass filter')
plt.xlabel('Eigenvalue (frequency))')
plt.ylabel('Filter value')
plt.show()

# plotting the filtered graph fourier transform
plt.plot(eigenvalues,highpassfiltered_GFT, color='k')
plt.title('Low pass filtered graph fourier transform of z, c='+ str(c))
plt.xlabel('Eigenvalue (frequency))')
plt.ylabel('$\mathcal{GF}(z)$')
plt.show()

# plotting the filtered image and the original image
plt.subplot(1,2,1)
plt.scatter(X[:,1], X[:,0], c=highpassfiltered_Z, cmap='gray')
plt.axis('off')
plt.ylim(600, 0)
plt.title('High pass filtered data, c='+ str(c))
plt.subplot(1,2,2)
plt.scatter(X[:,1], X[:,0], c=Z, cmap='gray')
plt.axis('off')
plt.ylim(600, 0)
plt.title('Original data')
plt.show()


"""---Task 4.8---"""
# filtering the original image (coffee.png) with a low-pass filter and a high-pass filter to compare the results with the results from task 4.6 and 4.7
coffee = plt.imread('Homeexam supplementary data\coffee.png') # loading the image
coffee_ft = np.fft.fftshift(np.fft.fft2(coffee)) # computing the fourier transform of the image

# creating the filter
filter = np.copy(coffee_ft)
c=50
for i in range(len(filter)):
    for j in range(len(filter[0])):
        if np.sqrt((i-len(filter)/2)**2+(j-len(filter[0])/2)**2) < c:
            filter[i][j] = 1
        else:
            filter[i][j] = 0

#plotting the filter
plt.imshow(np.abs(filter), cmap='gray')
plt.axis('off')
plt.title('Low pass filter, c='+ str(c))
plt.show()

highpass_coffee_ft = coffee_ft*(1-filter) # highpassfiltering in the frequency domain
lowpass_coffee_ft = coffee_ft*filter # lowpassfiltering in the frequency domain
highpass_coffee = np.fft.ifft2(np.fft.ifftshift(highpass_coffee_ft)) # computing the inverse fourier transform of the highpassfiltered image
lowpass_coffee = np.fft.ifft2(np.fft.ifftshift(lowpass_coffee_ft)) # computing the inverse fourier transform of the lowpassfiltered image

# plotting the results
plt.imshow(np.abs(lowpass_coffee), cmap='gray')
plt.axis('off')
plt.title('Low pass filtered coffee, c='+ str(c))
plt.show()
plt.imshow(np.abs(highpass_coffee), cmap='gray')
plt.axis('off')
plt.title('Low pass filtered coffee, c='+ str(c))
plt.show()
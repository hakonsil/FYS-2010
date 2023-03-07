import numpy as np
import matplotlib.pyplot as plt

#importing the data files
data1 = np.load(r'Homeexam supplementary data\point_cloud.npz')
data2 = np.load(r'Homeexam supplementary data\point_cloud_Laplacian.npz')

#extracting the data into arrays
X = data1['X'] #coordinates
Z = data1['Z'] #greyscale values
L = data2['L'] #Laplacian

"""Task 4.1"""
# creating a scatter plot of the data
plt.scatter(X[:,1], X[:,0], c=Z, cmap='gray')
plt.ylim(400, 0)
plt.show()

"""Task 4.2"""
#plot histogram of Z
plt.hist(Z, bins=100, color='k')
plt.title('Histogram of z')
plt.xlabel('Pixel value')
plt.ylabel('Pixel count')
plt.show()

"""Task 4.3"""
# eigenvectors are the same as fourier modes
# eigenvalues are the same as the 'frequency'
# so I will call them as such in the script
# the eigenvalues and eigenvectors can easily be found from L
# eigenvalues are sorted in ascending order by default from np.linalg.eigh()

# findin the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(L)
print(eigenvectors.shape)
# plotting the eigenvalues
plt.plot(eigenvalues, color='k')
plt.title('Eigenvalues')
plt.xlabel('Eigenvalue number')
plt.ylabel('Eigenvalue')
plt.show()

# check if we have all positive values
if np.min(eigenvalues) < 0:
    print('The eigenvalues are not all positive...')
elif np.min(eigenvalues) > 0:
    print('The eigenvalues are all positive!')
#check if the eigenvalues are sorted
for i in range(len(eigenvalues)-1):
    if eigenvalues[i] > eigenvalues[i+1]:
        print('The eigenvalues are not sorted...')
        break

"""Task 4.4"""

plt.scatter(X[:,0], X[:,1], c=eigenvectors[300], cmap='gray')
plt.axis('off')
plt.ylim(600, 0)
plt.title('Eigenvalue number: 1')
plt.show()


"""Task 4.5"""
# the graphs fourier transform is the inner product of the eigenvectors and the data (Z)
# this can be computed by multiplying the transpose of the eigenvectors with Z
# the result is a vector with the same length as Z 
# I couldn't find this in the lecture notes, so I used the definition from https://en.wikipedia.org/wiki/Graph_Fourier_transform
# the gft at each frequency is the inner product of the eigenvectors corresponding to that frequency and the data(Z)

GFT = eigenvectors.T @ Z # computing the graph fourier transform
# plotting
plt.plot(eigenvalues, GFT, color='k')
plt.title('Graph fourier transform of z')
plt.xlabel('Eigenvalue (frequency))')
plt.ylabel('$\mathcal{GF}(z)$')
plt.show()
# double checking the dimensions of GFT
print(GFT.shape)


"""Task 4.6"""
#the easiest way to implement an ideal low pass filter is to create a unit step function 
# where the cutoff frequency is the threshold, and then multiply the gft with the unit step function
# then I can take the igft to get the filtered data

# creating the unit step function

# creating the filter
c=3.5 # cutoff frequency

filter = np.copy(eigenvalues)
for i in range(len(filter)):
    if filter[i] < c:
        filter[i] = 1
    else:
        filter[i] = 0

lowpassfiltered_GFT = GFT*filter # filtering the gft
plt.plot(filter)
plt.title('Low pass filter')
plt.xlabel('Eigenvalue (frequency))')
plt.ylabel('Filter value')
plt.show()

plt.plot(eigenvalues,lowpassfiltered_GFT, color='k')
plt.title('Low pass filtered graph fourier transform of z, c='+ str(c))
plt.xlabel('Eigenvalue (frequency))')
plt.ylabel('$\mathcal{GF}(z)$')
plt.show()

# computing the inverse graph fourier transform
# again using the wikipedia definition
# the igft is the inner product of the eigenvectors(not transposed) and the filtered gft
# the result is a vector with the same size as Z

lowpassfiltered_Z = eigenvectors @ lowpassfiltered_GFT

# plotting the filtered data
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
# we can see that the low pass filter removes the high frequency noise when c = 3.5


c=0.3 # cutoff frequency

filter = np.copy(eigenvalues)
for i in range(len(filter)):
    if filter[i] < c:
        filter[i] = 1
    else:
        filter[i] = 0

highpassfiltered_GFT = GFT*(1-filter) # filtering the gft

highpassfiltered_Z = eigenvectors @ highpassfiltered_GFT

plt.plot(1-filter)
plt.title('High pass filter')
plt.xlabel('Eigenvalue (frequency))')
plt.ylabel('Filter value')
plt.show()

plt.plot(eigenvalues,highpassfiltered_GFT, color='k')
plt.title('Low pass filtered graph fourier transform of z, c='+ str(c))
plt.xlabel('Eigenvalue (frequency))')
plt.ylabel('$\mathcal{GF}(z)$')
plt.show()

# plotting the filtered data
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


coffee = plt.imread('Homeexam supplementary data\coffee.png')
coffee_ft = np.fft.fftshift(np.fft.fft2(coffee))
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
highpass_coffee_ft = coffee_ft*(1-filter)
lowpass_coffee_ft = coffee_ft*filter
highpass_coffee = np.fft.ifft2(np.fft.ifftshift(highpass_coffee_ft))
lowpass_coffee = np.fft.ifft2(np.fft.ifftshift(lowpass_coffee_ft))

plt.imshow(np.abs(lowpass_coffee), cmap='gray')
plt.axis('off')
plt.title('Low pass filtered coffee, c='+ str(c))
plt.show()
plt.imshow(np.abs(highpass_coffee), cmap='gray')
plt.axis('off')
plt.title('Low pass filtered coffee, c='+ str(c))
plt.show()
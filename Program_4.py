#Apply the frequency domain to the image high pass and low pass frequency domain . 

# IMPORTING THE LIBRARIES
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('modi.jpg', 0)    #reading the imagee in grayscale format

# applying the fast fourier transformation
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)

# getting the dimensions of the image
rows, cols = img.shape
crow, ccol = rows//2 , cols//2  


# LOW PASS FILTER

lpf_mask = np.zeros((rows, cols), np.uint8)
r = 50  
for i in range(rows):
    for j in range(cols):
        if (i - crow)**2 + (j - ccol)**2 <= r*r:
            lpf_mask[i, j] = 1

lpf_dft = dft_shift * lpf_mask
lpf_ishift = np.fft.ifftshift(lpf_dft)
lpf_img = np.fft.ifft2(lpf_ishift)
lpf_img = np.abs(lpf_img)


# HIGH PASS FILTER

hpf_mask = np.ones((rows, cols), np.uint8)
for i in range(rows):
    for j in range(cols):
        if (i - crow)**2 + (j - ccol)**2 <= r*r:
            hpf_mask[i, j] = 0

hpf_dft = dft_shift * hpf_mask
hpf_ishift = np.fft.ifftshift(hpf_dft)
hpf_img = np.fft.ifft2(hpf_ishift)
hpf_img = np.abs(hpf_img)

plt.figure(figsize=(10,8))

plt.subplot(2,2,1), plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(2,2,2), plt.imshow(lpf_img, cmap='gray')
plt.title('Low Pass Filter (Blurred)')

plt.subplot(2,2,3), plt.imshow(hpf_img, cmap='gray')
plt.title('High Pass Filter (Edges)')

plt.tight_layout()
plt.show()

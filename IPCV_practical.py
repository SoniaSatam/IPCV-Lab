#!/usr/bin/env python
# coding: utf-8

# In[22]:


import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


pip install --upgrade tensorflow


# In[23]:


from tensorflow.keras.datasets import mnist


# In[135]:


from tensorflow.keras.datasets import cifar10


# In[77]:


from tensorflow.keras.datasets import fashion_mnist


# In[136]:


(_, _), (x_test, _) = cifar10.load_data()
c_image1 = (x_test[0])
c_image2 = (x_test[1])


# In[24]:


def show_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


# In[81]:


(_, _), (x_test, _) = fashion_mnist.load_data()

# Sample image from MNIST dataset
fimage = x_test[1]  # Use the first image for demonstration
show_image(image, "image")


# In[148]:


(_, _), (x_test, _) = mnist.load_data()

# Sample image from MNIST dataset
image2 = x_test[0] 
image2 = x_test[9]  # Use the first image for demonstration
show_image(image1, "image1")
show_image(image2, "image2")


# In[134]:


def crop_image(image, x, y):
    height,width= image.shape
    cropped_image = image[y:y+height, x:x+width]
    return cropped_image

cropped_image = crop_image(image,10,20)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cropped_image, cmap='gray')
plt.title('Cropped Image')
plt.axis('off')

plt.show()


# In[121]:


def crop_image(image, x, y, width, height):
    cropped_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if 0 <= y + i < image.shape[0] and 0 <= x + j < image.shape[1]:
                cropped_image[i, j] = image[y + i, x + j]
    return cropped_image

# Select a random image from the MNIST dataset
random_index = np.random.randint(0, len(x_test))
image = x_test[random_index]

# Define the coordinates and size of the crop
x, y = 10, 20  # Top-left corner coordinates
width, height = 20, 20  # Width and height of the crop

# Crop the image
cropped_image = crop_image(image, x, y, width, height)

# Display the original and cropped images
# plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cropped_image, cmap='gray')
plt.title('Cropped Image')
plt.axis('off')

plt.show()


# In[172]:


def rgb_to_grayscale(image):
  height,width,channels = image.shape
  grayscaleImage = np.zeros((height,width))
  for x in range(height):
    for y in range(width):
        pixelValue = sum(image[x,y])/channels
        grayscaleImage[x,y] = pixelValue
  return grayscaleImage

grayscaled=rgb_to_grayscale(c_image1)
plt.imshow(grayscaled,cmap='gray')


# In[137]:


def addImages(image1, image2):
    height, width, channels = image1.shape
    addedImage = image1.copy()
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                addedImage[y, x, c] = min(int(image1[y, x, c]) + int(image2[y, x, c]), 255)
    return addedImage

added_image = addImages(c_image1,c_image2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(c_image1, cmap='gray')
plt.title('Image 1')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(c_image2, cmap='gray')
plt.title('Image 2')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(added_image,cmap='gray')
plt.title('Added Image')
plt.axis('off')


# In[139]:


def multiplyImages(image1, image2):
    assert image1.shape == image2.shape, "Images must have the same size"
    height, width, channels = image1.shape
    multipliedImage = image1.copy()
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                multipliedImage[y, x, c] = min(int(image1[y, x, c]) * int(image2[y, x, c]), 255)
    return multipliedImage

multiplied=multiplyImages(c_image1,c_image2)
plt.subplot(1, 3, 1)
plt.imshow(c_image1, cmap='gray')
plt.title('Image 1')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(c_image2, cmap='gray')
plt.title('Image 2')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(multiplied, cmap='gray')
plt.title('Multiplied Image')
plt.axis('off')


# In[140]:


def subtractImages(image1, image2):
    assert image1.shape == image2.shape, "Images must have the same size"
    height, width, channels = image1.shape
    subtractedImage = image1.copy()
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                subtractedImage[y, x, c] = max(int(image1[y, x, c]) - int(image2[y, x, c]), 0)
    return subtractedImage

subtracted =subtractImages(c_image1,c_image2)

plt.subplot(1, 3, 1)
plt.imshow(c_image1, cmap='gray')
plt.title('Image 1')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(c_image2, cmap='gray')
plt.title('Image 2')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(subtracted , cmap='gray')
plt.title('Subtracted  Image')
plt.axis('off')


# In[142]:


def divideImages(image1, image2):
    assert image1.shape == image2.shape, "Images must have the same size"
    height, width, channels = image1.shape
    dividedImage = image1.copy()
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                if int(image2[y, x, c]) != 0:
                    dividedImage[y, x, c] = min(int(image1[y, x, c]) / int(image2[y, x, c]), 255)
                else:
                    dividedImage[y, x, c] = 255  # Handle division by zero
    return dividedImage 

divided = divide_images(c_image1,c_image2)

plt.subplot(1, 3, 1)
plt.imshow(c_image1, cmap='gray')
plt.title('Image 1')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(c_image2, cmap='gray')
plt.title('Image 2')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(divided, cmap='gray')
plt.title('Divided Image')
plt.axis('off')


# In[51]:


#Arithmetic Blending Operations

import numpy as np
import matplotlib.pyplot as plt

# Define the arithmetic blending operations

# Addition: Image 1 + Image 2
def add_images(image1, image2):
    # Ensure both images have the same data type (dtype)
    image1 = image1.astype(np.uint16)
    image2 = image2.astype(np.uint16)
    
    # Add pixel values and clip to ensure the values are within 0-255 range
    added_image = np.clip(image1 + image2, 0, 255).astype(np.uint8)
    return added_image

# Subtraction: Image 1 - Image 2
def subtract_images(image1, image2):
    # Ensure both images have the same data type (dtype)
    image1 = image1.astype(np.int16)
    image2 = image2.astype(np.int16)
    
    # Subtract pixel values and clip to ensure the values are within 0-255 range
    subtracted_image = np.clip(image1 - image2, 0, 255).astype(np.uint8)
    return subtracted_image

# Multiplication: Image 1 * Image 2
def multiply_images(image1, image2):
    # Ensure both images have the same data type (dtype)
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    
    # Multiply pixel values and clip to ensure the values are within 0-255 range
    multiplied_image = np.clip((image1 * image2) / 255, 0, 255).astype(np.uint8)
    return multiplied_image

# Division: Image 1 / Image 2
def divide_images(image1, image2):
    # Ensure both images have the same data type (dtype)
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    
    # Avoid division by zero
    image2[image2 == 0] = 1
    
    # Divide pixel values
    divided_image = np.clip((image1 / image2) * 255, 0, 255).astype(np.uint8)
    return divided_image

# Randomly select two images from your dataset
random_indices = np.random.randint(0, len(x_test), size=2)
image1 = x_test[random_indices[0]]
image2 = x_test[random_indices[1]]

# Perform arithmetic blending operations
added_image = add_images(image1, image2)
subtracted_image = subtract_images(image1, image2)
multiplied_image = multiply_images(image1, image2)
divided_image = divide_images(image1, image2)

# Display the results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image1, cmap='gray')
plt.title('Image 1')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(image2, cmap='gray')
plt.title('Image 2')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(added_image, cmap='gray')
plt.title('Added Image')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(subtracted_image, cmap='gray')
plt.title('Subtracted Image')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(multiplied_image, cmap='gray')
plt.title('Multiplied Image')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(divided_image, cmap='gray')
plt.title('Divided Image')
plt.axis('off')

plt.show()


# In[143]:


def logical_and(image1, image2):
    height, width = len(image1), len(image1[0])
    result = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            result[i][j] = image1[i][j] & image2[i][j]
    return result
and_image = logical_and(c_image1,c_image2)
show_image(and_image, "and_image")


# In[150]:


def logical_or(image1, image2):
    height, width = len(image1), len(image1[0])
    result = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            result[i][j] = image1[i][j] | image2[i][j]
    return result
ans = logical_or(image1,image2)
show_image(ans,"OR image")


# In[151]:


def logical_not(image):
    height, width = len(image), len(image[0])
    result = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            result[i][j] = 255 - image[i][j]
    return result

notimg = logical_not(image1)
show_image(notimg,"NOT image")


# In[152]:


def logical_xor(image1, image2):
    height, width = len(image1), len(image1[0])
    result = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            result[i][j] = image1[i][j] ^ image2[i][j]
    return result
xor_image = logical_xor(image1, image2)
plt.imshow(xor_image, cmap='gray')
plt.title('Logical XOR Image')
plt.axis('off')
plt.show()


# In[153]:


def logical_nor(image1, image2):
    height, width = len(image1), len(image1[0])
    result = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            result[i][j] = 255 - (image1[i][j] | image2[i][j])  # NOR is the negation of OR
    return result

nor_image = logical_nor(image1, image2)

# Display the NOR image
plt.imshow(nor_image, cmap='gray')
plt.title('Logical NOR Image')
plt.axis('off')
plt.show()


# In[60]:


def left_shift(image, shift):
    # Perform bitwise left shift operation
    shifted_image = np.left_shift(image, shift)
    return shifted_image
shift_value = 2

# Perform bitwise left shift operation on the image
shifted_image = left_shift(image1, shift_value)

# Display the original and shifted images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(shifted_image, cmap='gray')
plt.title('Left Shifted Image')
plt.axis('off')

plt.show()


# In[61]:


def right_shift(image, shift):
    # Perform bitwise right shift operation
    shifted_image = np.right_shift(image, shift)
    return shifted_image
shift_value = 2

# Perform bitwise right shift operation on the image
shifted_image = right_shift(image1, shift_value)

# Display the original and shifted images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(shifted_image, cmap='gray')
plt.title('Right Shifted Image')
plt.axis('off')

plt.show()


# In[62]:


def threshold(image, threshold_value):
    # Apply thresholding
    image = np.array(image)
    thresholded_image = np.where(image >= threshold_value, 255, 0)
    return thresholded_image

thresholded = threshold(image1, 50)
show_image(thresholded, "threshold")


# In[ ]:


def gray_level_slicing_with_background(image,a,b):
    grey_img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,w,_ = image.shape
    wbg_img = np.zeros((h, w), np.uint8)
    for i in range(h):
      for j in range(w):
        r = grey_img1[i][j]
        if a <= r <= b:
            wbg_img[i][j] = 255
        else:
            wbg_img[i][j] = r
    return wbg_img

low = 100
high = 200


# In[155]:


def gray_level_slicing(image, low, high, background_intensity=None):
    # Create a mask for pixels within the specified intensity range
    mask = (image >= low) & (image <= high)
    
    # If background intensity is specified, set pixels outside the range to background intensity
    if background_intensity is not None:
        result = np.where(mask, 255, background_intensity)
    else:
        result = np.where(mask, 255, 0)
    
    return result.astype(np.uint8)


# Define the intensity range for gray level slicing
low = 100
high = 200

# Perform gray level slicing without background intensity slicing
result_without_background = gray_level_slicing(c_image1, low, high)

# Perform gray level slicing with background intensity slicing
background_intensity = 128
result_with_background = gray_level_slicing(c_image1, low, high, background_intensity)

# Display the original and processed images
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(c_image1, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(result_without_background, cmap='gray')
plt.title('Gray Level Slicing (No Background)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(result_with_background, cmap='gray')
plt.title('Gray Level Slicing (With Background)')
plt.axis('off')

plt.show()


# In[157]:


plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.hist(c_image1.flatten(), bins=256, color='blue', alpha=0.5)
plt.title('Original Image Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(result_without_background.flatten(), bins=256, color='green', alpha=0.5)
plt.title('Gray Level Slicing Histogram (No Background)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.hist(c_image1.flatten(), bins=256, color='blue', alpha=0.5)
plt.hist(result_with_background.flatten(), bins=256, color='red', alpha=0.5)
plt.title('Original and Gray Level Slicing Histograms (With Background)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend(['Original Image', 'Gray Level Slicing (With Background)'])

plt.tight_layout()
plt.show()


# In[72]:


def contrast_stretching(image):
    # Define the contrast stretching parameters
    #(r1,s1)=(min, 0)
    #(r2,s2)=(max, 255)

    r1 = np.min(image1)
    s1 = 0
    r2 = np.max(image1)
    s2 = 255
    pixelVal_vec = np.vectorize(lambda pixel: s1 + (pixel - r1) * (s2 - s1) / (r2 - r1))
    contrast_stretched = pixelVal_vec(image1)
    return contrast_stretched.astype(np.uint8)


# Apply contrast stretching to the cropped image

contrast_stretched = contrast_stretching(image1)


# Display the original and stretched images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(contrast_stretched, cmap='gray')
plt.title('Contrast-Stretched Image')
plt.axis('off')

plt.show()


# In[73]:


import matplotlib.pyplot as plt

def plot_histogram(image,title):
    # Plot the histogram of image
    plt.figure()
    plt.hist(image.ravel(), bins=256, range=(0, 256))
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {title}")
    plt.show()


# Plot the histogram of contrast stretched image
plot_histogram(cropped_image,"cropped_image")
plot_histogram(contrast_stretched,"contrast_stretched")


# In[82]:


def power_law_transform(image, gamma):
    # Normalize pixel values to range [0, 1]
    normalized_image = image / 255.0
    # Apply power law transformation
    transformed_image = np.power(normalized_image, gamma)
    # Scale the transformed image back to range [0, 255]
    transformed_image = (transformed_image * 255).astype(np.uint8)
    return transformed_image

powerlaw = power_law_transform(fimage,1.2)
show_image(fimage, "image1")
show_image(powerlaw, "powerlaw")


# In[83]:


def log_transform(image):
    # Normalize pixel values to range [0, 1]
    normalized_image = image / 255.0
    # Apply log transformation
    transformed_image = np.log(1 + normalized_image)
    # Scale the transformed image back to range [0, 255]
    transformed_image = (transformed_image * 255).astype(np.uint8)
    return transformed_image
ans = log_transform(fimage)
show_image(fimage,"fimage")
show_image(ans,"ans")


# In[87]:


def add_gaussian_noise(image, mean=0, std=25):
    # Generate Gaussian noise with the same shape as the input image
    noise = np.random.normal(mean, std, image.shape)
    
    # Add the noise to the image
    noisy_image = image + noise
    
    # Clip pixel values to ensure they are within the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    
    # Convert the image back to the appropriate data type
    noisy_image = noisy_image.astype(np.uint8)
    
    return noisy_image


# Add Gaussian noise to the image
noisy_image = add_gaussian_noise(image1)

# Display the original and noisy images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image1,cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(noisy_image,cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.show()


# In[92]:


def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = np.copy(image)
    
    # Generate random indices for salt noise
    salt_indices = np.random.rand(*image.shape) < salt_prob
    noisy_image[salt_indices] = 255
    
    # Generate random indices for pepper noise
    pepper_indices = np.random.rand(*image.shape) < pepper_prob
    noisy_image[pepper_indices] = 0
    
    return noisy_image


# Add salt and pepper noise to the image
noisy_image = add_salt_and_pepper_noise(fimage)

# Display the original and noisy images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(fimage,cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image with Salt & Pepper Noise')
plt.axis('off')

plt.show()


# In[93]:


def median_filter(image, kernel_size=3):
    rows, cols = image.shape
    result = image.copy()

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighborhood = [image[i + di][j + dj] for di in range(-1, 2) for dj in range(-1, 2)]
            result[i][j] = sorted(neighborhood)[len(neighborhood) // 2]

    return result

# Function to perform mean filtering on the image
def mean_filter(image, kernel_size=3):
    rows, cols = image.shape
    result = image.copy()

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighborhood_sum = sum(image[i + di][j + dj] for di in range(-1, 2) for dj in range(-1, 2))
            result[i][j] = neighborhood_sum // 9

    return result

# Perform median filtering
median_filtered_image = median_filter(image)

# Perform mean filtering
mean_filtered_image = mean_filter(image)

# Plot original, median filtered, and mean filtered images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(median_filtered_image, cmap='gray')
plt.title('Median Filtered Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mean_filtered_image, cmap='gray')
plt.title('Mean Filtered Image')
plt.axis('off')

plt.show()


# In[98]:


# @title Gaussian low pass and High pass filtering
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(_, _), (x_test, _) = mnist.load_data()

# Sample image from MNIST dataset
image = x_test[0]  # Use the first image for demonstration

# Function to perform 2D convolution
def convolution2d(image, kernel):
    # Get dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Create padded image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Initialize result matrix
    result = np.zeros_like(image)

    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            result[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return result

# Function to generate 2D Gaussian kernel
def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

# Function for Gaussian low pass filtering
def gaussian_low_pass_filter(image, size, sigma):
    kernel = gaussian_kernel(size, sigma)
    return convolution2d(image, kernel)

# Function for Gaussian high pass filtering
def gaussian_high_pass_filter(image, size, sigma):
    low_pass_filtered_image = gaussian_low_pass_filter(image, size, sigma)
    high_pass_filtered_image = image - low_pass_filtered_image
    return high_pass_filtered_image

# Parameters for Gaussian low pass and high pass filtering
size = 5  # Size of the Gaussian kernel (odd integer)
sigma = 1  # Standard deviation of the Gaussian distribution

# Perform Gaussian low pass filtering
gaussian_low_pass_filtered_image = gaussian_low_pass_filter(image, size, sigma)

# Perform Gaussian high pass filtering
gaussian_high_pass_filtered_image = gaussian_high_pass_filter(image, size, sigma)

# Plot original and filtered images
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gaussian_low_pass_filtered_image, cmap='gray')
plt.title('Gaussian Low Pass Filtered Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gaussian_high_pass_filtered_image, cmap='gray')
plt.title('Gaussian High Pass Filtered Image')
plt.axis('off')
plt.show()


# In[99]:


# @title High Boost Filtering

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
# Load MNIST dataset
(_, _), (x_test, _) = mnist.load_data()

# Sample image from MNIST dataset
image = x_test[0]  # Use the first image for demonstration

# Function to perform high boost filtering
def high_boost_filtering(image, kernel_size, boost_factor):
    # Get dimensions of the image
    image_height, image_width = image.shape

    # Initialize result matrix
    result = np.zeros_like(image)

    # Define high-pass filter kernel
    high_pass_kernel = [[-1] * kernel_size] * kernel_size
    high_pass_kernel[kernel_size // 2][kernel_size // 2] = kernel_size ** 2 - 1

    # Pad the image
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')

    # Perform high-pass filtering
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = np.sum(np.multiply(region, high_pass_kernel))

    # Calculate the high boost filtered image
    high_boost_filtered_image = image + boost_factor * result

    return high_boost_filtered_image

# Parameters for high boost filtering
kernel_size = 3  # Size of the high-pass filter kernel (odd integer)
boost_factor = 1.5  # Boost factor for enhancing high-frequency components

# Perform high boost filtering
high_boost_filtered_image = high_boost_filtering(image, kernel_size, boost_factor)

# Plot original and high boost filtered images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(high_boost_filtered_image, cmap='gray')
plt.title('High Boost Filtered Image')
plt.axis('off')

plt.show()


# In[102]:


# Function to compute histogram of an image
import numpy as np
import matplotlib.pyplot as plt

# Function to compute histogram of an image
def compute_histogram(image):
    histogram = [0] * 256  # Initialize histogram with 256 bins
    for row in image:
        for pixel in row:
            histogram[pixel] += 1
    return histogram

# Function to compute cumulative distribution function (CDF) of a histogram
def compute_cdf(histogram):
    cdf = [0] * 256  # Initialize CDF with 256 bins
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histogram[i]
    return cdf

# Function for histogram equalization
def histogram_equalization(image):
    rows, cols = image.shape  # For grayscale image, shape will have only 2 dimensions
    num_pixels = rows * cols
    histogram = compute_histogram(image)
    cdf = compute_cdf(histogram)
    equalized_image = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            equalized_image[i, j] = int((cdf[image[i, j]] / num_pixels) * 255)
    return equalized_image

# Load an example grayscale image (replace 'example_image.jpg' with the path to your image)


# Convert the image to grayscale if it's a color image
if len(image.shape) > 2:
    image = np.mean(image, axis=2).astype(np.uint8)

# Perform histogram equalization
equalized_image = histogram_equalization(image)

# Plot original and equalized images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.show()


# In[105]:


# @title Ideal low pass and Butterworth low pass filtering
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(_, _), (x_test, _) = mnist.load_data()

# Sample image from MNIST dataset
image = x_test[0]  # Use the first image for demonstration

# Function to compute 2D Discrete Fourier Transform (DFT)
def dft2(image):
    M, N = image.shape
    dft = np.zeros((M, N), dtype=np.complex128)
    for u in range(M):
        for v in range(N):
            sum_val = 0
            for x in range(M):
                for y in range(N):
                    sum_val += image[x, y] * np.exp(-2j * np.pi * ((u * x) / M + (v * y) / N))
            dft[u, v] = sum_val
    return dft

# Function to compute 2D Inverse Discrete Fourier Transform (IDFT)
def idft2(dft):
    M, N = dft.shape
    idft = np.zeros((M, N), dtype=np.complex128)
    for x in range(M):
        for y in range(N):
            sum_val = 0
            for u in range(M):
                for v in range(N):
                    sum_val += dft[u, v] * np.exp(2j * np.pi * ((u * x) / M + (v * y) / N))
            idft[x, y] = sum_val / (M * N)
    return idft

# Function to generate Ideal Low Pass Filter
def ideal_low_pass_filter(image_shape, cutoff_freq):
    rows, cols = image_shape
    center_row, center_col = rows // 2, cols // 2
    filter_matrix = np.zeros(image_shape)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if dist <= cutoff_freq:
                filter_matrix[i, j] = 1
    return filter_matrix

# Function to generate Butterworth Low Pass Filter
def butterworth_low_pass_filter(image_shape, cutoff_freq, order):
    rows, cols = image_shape
    center_row, center_col = rows // 2, cols // 2
    filter_matrix = np.zeros(image_shape)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            filter_matrix[i, j] = 1 / (1 + (dist / cutoff_freq) ** (2 * order))
    return filter_matrix

# Function for applying frequency domain filtering
def frequency_domain_filtering(image, filter_matrix):
    # Compute 2D DFT of the image
    image_dft = dft2(image)
    # Apply the filter in frequency domain
    filtered_image_dft = image_dft * filter_matrix
    # Compute the inverse 2D DFT
    filtered_image = np.abs(idft2(filtered_image_dft))
    return filtered_image

# Parameters for Ideal Low Pass Filter
ideal_cutoff_freq = 30  # Adjust cutoff frequency as needed

# Parameters for Butterworth Low Pass Filter
butterworth_cutoff_freq = 30  # Adjust cutoff frequency as needed
butterworth_order = 2  # Adjust order as needed

# Generate Ideal Low Pass Filter
ideal_lp_filter = ideal_low_pass_filter(image.shape, ideal_cutoff_freq)

# Generate Butterworth Low Pass Filter
butterworth_lp_filter = butterworth_low_pass_filter(image.shape, butterworth_cutoff_freq, butterworth_order)

# Apply Ideal Low Pass Filtering
ideal_lp_filtered_image = frequency_domain_filtering(image, ideal_lp_filter)

# Apply Butterworth Low Pass Filtering
butterworth_lp_filtered_image = frequency_domain_filtering(image, butterworth_lp_filter)

# Plot original and filtered images
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(ideal_lp_filter, cmap='gray')
plt.title('Ideal LP Filter')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(butterworth_lp_filter, cmap='gray')
plt.title('Butterworth LP Filter')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(ideal_lp_filtered_image, cmap='gray')
plt.title('Ideal LP Filtered Image')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(butterworth_lp_filtered_image, cmap='gray')
plt.title('Butterworth LP Filtered Image')
plt.axis('off')

plt.show()


# In[106]:


def erosion(image):
    rows, cols = image.shape
    result = image.copy()

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if all(image[i + di][j + dj] > 0 for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                result[i][j] = 1
            else:
                result[i][j] = 0

    return result


def dilation(image):

    rows, cols = image.shape
    result = image.copy()

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if any(image[i + di][j + dj] >0 for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                result[i][j] = 1

    return result

show_image(image,"mnist_image1")

ans = dilation(image)
show_image(ans,"ans")

ans2 = erosion(image)
show_image(ans2,"ans2")


# In[109]:


def opening(image):
    return dilation(erosion(image))

# Function to perform closing operation on the image
def closing(image):
    return erosion(dilation(image))

ans = opening(image)
show_image(ans,"Opening")

ans2 = closing(image)
show_image(ans2,"Closing")


# In[110]:


se=[[1,1],[1,0]]
seed=255

#first convert greyscale to a binary image
bi_img=np.zeros_like(image,dtype=np.uint8)
t=150
for i in range(image.shape[0]):
  for j in range (image.shape[1]):
    pix=image[i,j]
    if(pix>=t):
      bi_img[i,j]=255
    else:
      bi_img[i,j]=0

#dilation
row,col=bi_img.shape
dil=np.zeros_like(bi_img)
for i in range(row-1):
  for j in range(col-1):
    if(bi_img[i][j]==seed):
      dil[i][j]=255
      dil[i+1][j]=255
      dil[i][j+1]=255

#erosion
row,col=bi_img.shape
ero=np.zeros_like(bi_img)
for i in range(row-1):
  for j in range(col-1):
    if(bi_img[i][j]==bi_img[i+1][j]==bi_img[i][j+1]==seed):
      ero[i][j]=255

plt.subplot(1, 3, 1)
plt.imshow(bi_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(dil, cmap='gray')
plt.title('Dilation')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(ero, cmap='gray')
plt.title('Erosion')
plt.axis('off')
plt.show()


# In[113]:


#hit
hit=np.zeros_like(bi_img)
for i in range(row-1):
  for j in range(col-1):
    if(bi_img[i][j]==bi_img[i+1][j]==bi_img[i][j+1]==seed and bi_img[i+1][j+1]==0):
      hit[i][j]=255

#miss
miss=np.ones_like(bi_img)
for i in range(row-1):
  for j in range(col-1):
    if(bi_img[i][j]==bi_img[i+1][j]==bi_img[i][j+1]==seed and bi_img[i+1][j+1]==0):
      miss[i][j]=0
    
plt.subplot(1, 3, 1)
plt.imshow(bi_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(hit, cmap='gray')
plt.title('Hit')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(miss,cmap='gray')
plt.title('Miss')
plt.axis('off')
plt.show()   


# In[159]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(train_images, _), _ = fashion_mnist.load_data()

def region_growing(image, seed, threshold):
    image = image.astype(np.int64)
    height, width = image.shape
    visited = np.zeros((height, width), dtype=bool)
    segmented_image = np.zeros_like(image)

    queue = [seed]
    while queue:
        current_point = queue.pop(0)
        x, y = current_point

        if not visited[x, y]:
            if abs(image[x, y] - np.mean(segmented_image)) < threshold:
                segmented_image[x, y] = 255
                visited[x, y] = True
                if x > 0:
                    queue.append((x - 1, y))
                if x < height - 1:
                    queue.append((x + 1, y))
                if y > 0:
                    queue.append((x, y - 1))
                if y < width - 1:
                    queue.append((x, y + 1))

    return segmented_image
def region_splitting(image, threshold):
    image = image.astype(np.int64)
    regions = []
    height, width = image.shape
    visited = np.zeros((height, width), dtype=bool)

    def explore_region(start_point):
        region = []
        queue = [start_point]

        while queue:
            current_point = queue.pop(0)
            x, y = current_point

            if not visited[x, y]:
                visited[x, y] = True
                region.append(current_point)
                if x > 0 and abs(image[x, y] - image[x-1, y]) < threshold:
                    queue.append((x - 1, y))
                if x < height - 1 and abs(image[x, y] - image[x+1, y]) < threshold:
                    queue.append((x + 1, y))
                if y > 0 and abs(image[x, y] - image[x, y-1]) < threshold:
                    queue.append((x, y - 1))
                if y < width - 1 and abs(image[x, y] - image[x, y+1]) < threshold:
                    queue.append((x, y + 1))

        return region

    for i in range(height):
        for j in range(width):
            if not visited[i, j]:
                region = explore_region((i, j))
                regions.append(region)

    return regions
def region_merging(regions, threshold):
    merged_regions = []
    for region in regions:
        mean_intensity = np.mean([train_images[pixel[0], pixel[1]] for pixel in region])
        for merged_region in merged_regions:
            if abs(mean_intensity - np.mean([train_images[pixel[0], pixel[1]] for pixel in merged_region])) < threshold:
                merged_region.extend(region)
                break
        else:
            merged_regions.append(region)
    return merged_regions

random_indices = np.random.randint(0, train_images.shape[0], size=1)

plt.figure(figsize=(20, 30))
for i, idx in enumerate(random_indices):
    random_image = train_images[idx]
    seed_point = (random_image.shape[0] // 2, random_image.shape[1] // 2)
    threshold_rg = 20
    segmented_image_rg = region_growing(random_image, seed_point, threshold_rg)
    threshold_rs = 20
    regions = region_splitting(random_image, threshold_rs)
    merged_regions = region_merging(regions, threshold_rs)
    plt.subplot(15, 4, i * 4 + 1)
    plt.imshow(random_image, cmap='gray')
    plt.title('Original')

    plt.subplot(15, 4, i * 4 + 2)
    plt.imshow(segmented_image_rg, cmap='gray')
    plt.title('Region Growing')

    plt.subplot(15, 4, i * 4 + 3)
    plt.imshow(random_image, cmap='gray')
    for region in regions:
        plt.plot([pixel[1] for pixel in region], [pixel[0] for pixel in region], 'r')
    plt.title('Region Splitting')

    plt.subplot(15, 4, i * 4 + 4)
    plt.imshow(random_image, cmap='gray')
    for region in merged_regions:
        plt.plot([pixel[1] for pixel in region], [pixel[0] for pixel in region], 'g')
    plt.title('Region Merging')

plt.tight_layout()
plt.show()


# In[162]:


# Region growing function
def region_growing(image, seed, threshold):
    rows, cols = image.shape
    visited = np.zeros_like(image, dtype=bool)
    region = np.zeros_like(image, dtype=bool)

    def is_valid_pixel(x, y):
        return 0 <= x < rows and 0 <= y < cols and not visited[x][y]

    def is_similar_intensity(x, y):
        return abs(image[x][y] - image[seed[0]][seed[1]]) <= threshold

    def grow_region(x, y):
        if is_valid_pixel(x, y) and is_similar_intensity(x, y):
            visited[x][y] = True
            region[x][y] = True
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    grow_region(x + dx, y + dy)

    grow_region(seed[0], seed[1])
    return region

# Seed point for region growing, almost at the middle
seed_point = (14, 14)

# Threshold for intensity similarity
intensity_threshold = 20

# Perform region growing on cifar_image1, cause the given image is too large
grown_region = region_growing(image1, seed_point, intensity_threshold)
show_image(image1, "image1")
show_image(grown_region, "grown_region")


# In[163]:


# @title Region Growing, Merging, spliting
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

# Sample MNIST image
image = x_test[0]  # Assuming x_test is the MNIST dataset loaded earlier

# Region growing function
def region_growing(image, seed, threshold):
    rows, cols = image.shape
    visited = np.zeros_like(image, dtype=bool)
    region = np.zeros_like(image, dtype=bool)

    def is_valid_pixel(x, y):
        return 0 <= x < rows and 0 <= y < cols and not visited[x][y]

    def is_similar_intensity(x, y):
        return abs(image[x][y] - image[seed[0]][seed[1]]) <= threshold

    def grow_region(x, y):
        if is_valid_pixel(x, y) and is_similar_intensity(x, y):
            visited[x][y] = True
            region[x][y] = True
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    grow_region(x + dx, y + dy)

    grow_region(seed[0], seed[1])
    return region

# Seed point for region growing
seed_point = (14, 14)

# Threshold for intensity similarity
intensity_threshold = 20

# Perform region growing
grown_region = region_growing(image, seed_point, intensity_threshold)

# Label connected components
labeled_region, num_labels = label(grown_region)

# Display original image and grown region
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(labeled_region, cmap='nipy_spectral')
plt.title('Grown Region')
plt.axis('off')

plt.show()


# In[164]:


def histogram_stretching(image, r_min, r_max):
    min_pixel = np.min(image)
    max_pixel = np.max(image)
    
    stretched_image = (image - min_pixel) * ((r_max - r_min) / (max_pixel - min_pixel)) + r_min
    stretched_image = np.clip(stretched_image, 0, 255)
    
    return stretched_image.astype(np.uint8)

def histogram(image):
    hist = np.zeros(256, dtype=np.uint8)
    for pixel_value in image.flatten():
        hist[pixel_value] += 1
    return hist

hist_original = histogram(image1)
hist_stretched = histogram(stretched_image)

# Plot histograms
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(hist_original, color='b')
plt.title('Original Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.plot(hist_stretched, color='r')
plt.title('Stretched Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[173]:


#Histogram Equalization
img = np.asarray(grayscaled)

# put pixels in a 1D array by flattening out img array
flat = img.flatten()

# show the histogram
plot_histogram(img,"og img")


# In[175]:


def get_histogram(image, bins):
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)

    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[int(pixel)] += 1

    # return our final result
    return histogram

# execute our histogram function
hist = get_histogram(flat, 256)
print(hist)


# In[176]:


# create our cumulative sum function
def cumsum(a):
    cumulative_sum = []
    total = 0
    for element in a:
        total += element
        cumulative_sum.append(total)
    return cumulative_sum

# execute the fn
cs = cumsum(hist)
cs = np.array(cs)
# display the result
plt.plot(cs)


# In[177]:


# We now have the cumulative sum, but as you can see, the values are huge (> 6,000,000). We’re going to be matching these values to our original image in the final step, so we have to normalize them to conform to a range of 0–255. Here’s one last formula for us to code up:
# numerator & denomenator
nj = (cs - cs.min()) * 255
N = cs.max() - cs.min()

# re-normalize the cumsum
cs_normalized  = nj / N

# cast it back to uint8 since we can't use floating point values in images
cs_normalized  = cs_normalized.astype('uint8')

plt.plot(cs_normalized )


# In[178]:


# get the value from cumulative sum for every index in flat, and set that as img_new
img_new = cs[flat.astype(int)]
# put array back into original shape since we flattened it
img_new = np.reshape(img_new, img.shape)

# set up side-by-side image display
fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)

fig.add_subplot(1,2,1)
plt.imshow(grayscaled, cmap='gray')

# display the new image
fig.add_subplot(1,2,2)
plt.imshow(img_new, cmap='gray')

plt.show(block=True)


# In[179]:


plot_histogram(img_new, "img_new")


# In[180]:


def histogram_stretching(image):
    # Flatten the image
    image = np.array(image)
    flattened_image = image.flatten()

    # Find minimum and maximum pixel values
    min_value = np.min(flattened_image)
    max_value = np.max(flattened_image)

    # Calculate the range of pixel values
    pixel_range = max_value - min_value

    # Stretch the histogram to cover the full dynamic range [0, 255]
    stretched_image = ((flattened_image - min_value) / pixel_range) * 255

    # Reshape the stretched image to its original shape
    stretched_image = stretched_image.reshape(image.shape)

    return stretched_image

stretched = histogram_stretching(image1)
show_image(image1, "image")
show_image(stretched, "st")
plot_histogram(stretched,"stretched")


# In[ ]:





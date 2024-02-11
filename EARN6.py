import os
import time
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter


script_name = sys.argv[0]
imgFile = sys.argv[1] if len(sys.argv) > 1 else "image"
number_of_images = int(sys.argv[2] if len(sys.argv) > 1 else 3)

print(f"Processing file: {imgFile}.")




def open_image_crop_as_RGB(image_path):
    return open_image_crop_as_RGB(image_path, 1000)

def open_image_crop_as_RGB(image_path, cropLW):
    # Load the image
    img = Image.open(image_path)

    cropLw = cropLW/2 #500 by default
    
    # I control for the center of the image because edges are annoying
    box = (img.width/2 - cropLw, img.height/2 - cropLw, img.width/2 + cropLw, img.height/2 + cropLw) 
    img = img.crop(box)
    
    img = img.convert('RGB')

    # Convert the image to a numpy array
    RGB_array = np.array(img)

    return RGB_array


def RGB_noise_cull(RGB_array, target_wavelength):

    r_channel = RGB_array[:, :, 0].astype('int16')
    g_channel = RGB_array[:, :, 1].astype('int16')
    b_channel = RGB_array[:, :, 2].astype('int16')

    return np.clip(g_channel - (r_channel + b_channel)/2, 0, 255)


def dark_comparison_cull(array, dark_array): # static false pixel cull, assumes there is a dark image
    
	return np.clip(array - dark_array, 0, 255)


def multi_comparison_cull(imgs_array): # static false pixel cull, assumes there is a dark image

    num_images = len(imgs_array)

    summed_intensities = imgs_array[0]

    for image_num in range(1, num_images):
        summed_intensities = summed_intensities + imgs_array[image_num]

    mean_array = summed_intensities/num_images
    
    sum_of_differences = (imgs_array[0] - mean_array)*(imgs_array[0] - mean_array)

    for image_num in range(1, num_images):
        sum_of_differences = sum_of_differences + (imgs_array[image_num] - mean_array)*(imgs_array[image_num] - mean_array)
    
    standard_deviation_array = np.sqrt(sum_of_differences/3)

    tolerance = np.mean(standard_deviation_array)

    print(f"Tolerance set to average standard deviation: {tolerance}")

    standard_deviation_array[standard_deviation_array < tolerance] = 1 # if variance is within tolerance allow to pass
    
    standard_deviation_array[standard_deviation_array != 1] = 0 # ...else set to zero, reject

    output_array = standard_deviation_array*mean_array

    return output_array


def process_img_preblur(image_path, num_images, cropLW, gaussian_radius, target_wavelength):

    imgs_array = []
    dark_imgs_array = []

    for image_num in range(1, num_images + 1):

        current_image_path = f"{image_path[:image_path.rfind('_')]}_{image_num}.jpg"

        RGB_array = open_image_crop_as_RGB(current_image_path, cropLW)
        dark_RGB_array = open_image_crop_as_RGB(f"D{current_image_path[current_image_path.find('_'):]}", cropLW)

        intensity_array = RGB_noise_cull(RGB_array, target_wavelength)
        dark_intensity_array = RGB_noise_cull(dark_RGB_array, target_wavelength)

        imgs_array.append(gaussian_filter(intensity_array, sigma=1))
        dark_imgs_array.append(gaussian_filter(dark_intensity_array, sigma=1))

    if len(imgs_array) > 1:

        mean_intensity = multi_comparison_cull(imgs_array)
        mean_dark_intensity = multi_comparison_cull(dark_imgs_array)

    else:

        mean_intensity = imgs_array[0]
        mean_dark_intensity = dark_imgs_array[0]

    mean_intensity = gaussian_filter(mean_intensity, sigma=50)
    mean_dark_intensity = gaussian_filter(mean_dark_intensity, sigma=50)

    processed_array = dark_comparison_cull(mean_intensity, mean_dark_intensity)

    #processed_array[processed_array < .2] = 0

    return processed_array


def plot_surface(array):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    z = array

    y = np.arange(len(z))
    x = np.arange(len(z[0]))

    (x,y) = np.meshgrid(x,y)

    surf = ax.plot_surface(x,y,z, cmap=cm.coolwarm)
    ax.set_zlim(0, 2)
    fig.colorbar(surf, shrink=0.5, aspect=10)





def main():

    processed_img = process_img_preblur(imgFile, number_of_images, 1500, 30, 490)

    plot_surface(processed_img)
    print(f"Summed value for {imgFile}: {np.sum(processed_img)}")
    plt.show()

if __name__ == "__main__":
    main()
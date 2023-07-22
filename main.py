import cv2
import numpy as np


def add_gaussian_noise(image, mean=0, stddev=10):
    mean = 0
    std = 50
    noise = np.random.normal(mean, std, size=image.shape)
    noisy_image = image + noise
    return noisy_image


def svd(A, epsilon=1e-10):
    # Get the size of the matrix A
    m, n = A.shape

    # Compute A^T * A
    ATA = np.dot(A.T, A)

    # Compute the eigenvalues and eigenvectors of A^T * A
    eigenvalues, eigenvectors = np.linalg.eig(ATA)

    # Sort eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute the singular values
    singular_values = np.sqrt(eigenvalues)

    # Compute the left singular vectors
    U = np.dot(A, eigenvectors)
    U /= np.linalg.norm(U, axis=0)

    # Compute the right singular vectors
    V = eigenvectors

    # Remove singular values close to zero
    nonzero_mask = singular_values > epsilon
    singular_values = singular_values[nonzero_mask]
    U = U[:, nonzero_mask]
    V = V[:, nonzero_mask]

    return singular_values, U, V.T


def denoise_image(image, k):
    # Convert the image to a floating-point representation
    image = image.astype(float)

    # Separate color channels
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    # Apply SVD to each color channel
    red_singular_values, red_U, red_Vt = svd(red_channel)
    green_singular_values, green_U, green_Vt = svd(green_channel)
    blue_singular_values, blue_U, blue_Vt = svd(blue_channel)

    # Keep only the k largest singular values/vectors
    red_singular_values[k:] = 0
    green_singular_values[k:] = 0
    blue_singular_values[k:] = 0

    # Reconstruct the color channels
    red_denoised = np.dot(np.dot(red_U, np.diag(red_singular_values)), red_Vt)
    green_denoised = np.dot(np.dot(green_U, np.diag(green_singular_values)), green_Vt)
    blue_denoised = np.dot(np.dot(blue_U, np.diag(blue_singular_values)), blue_Vt)

    # Combine the color channels to form the denoised image
    denoised_image = np.stack((red_denoised, green_denoised, blue_denoised), axis=2)

    return denoised_image.astype(np.uint8)


# Load the colorful image
image = cv2.imread('image.jpg')

# Add Gaussian noise to the image
noisy_image = add_gaussian_noise(image, mean=0, stddev=0.1)

# Denoise the image with k=50 (adjust this parameter as desired)
denoised_image = denoise_image(np.array(noisy_image), k=50)

# Display the original and noisy images
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Denoised Image', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np


def apply_gamma_correction(original_image, gamma=2.2):
    """
    Apply gamma correction to an image.

    Args:
        original_image (numpy.ndarray): The input image.
        gamma (float): The gamma value for correction (default is 2.2).

    Returns:
        numpy.ndarray: The gamma-corrected image.
    """
    # Normalize the image by dividing all pixel values by 255.0
    image_normalized = original_image / 255.0

    # Apply gamma correction to the normalized image
    corrected_image = np.power(image_normalized, gamma)

    # Scale the corrected values back to the range [0, 255] and convert to uint8
    corrected_image = (corrected_image * 255).astype(np.uint8)

    return corrected_image

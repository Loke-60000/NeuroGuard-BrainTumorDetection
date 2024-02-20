import cv2
import numpy as np
import os

def auto_crop_and_center_image(image_path, output_path, canvas_width=256, canvas_height=256):
    """
    Automatically crops and centers an image around the region of interest.
    The region of interest is the largest contour found in the image.
    The image is resized to fit within a specified canvas size and centered.
    The final image is saved to the output path.

    :param image_path: The path to the input image.
    :param output_path: The path to save the processed image.
    :param canvas_width: The width of the canvas in pixels.
    :param canvas_height: The height of the canvas in pixels.
    """
    import cv2
    import numpy as np

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No contours found in image {image_path}. Image is not processed.")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    cropped_image = image[y:y+h, x:x+w]

    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_h = int(canvas_height / aspect_ratio)
        new_w = canvas_width
    else:
        new_w = int(canvas_width * aspect_ratio)
        new_h = canvas_height

    new_w, new_h = max(1, new_w), max(1, new_h)

    resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    background = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    x_offset = (canvas_width - new_w) // 2
    y_offset = (canvas_height - new_h) // 2

    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image

    cv2.imwrite(output_path, background)
    print(f"Processed and saved: {output_path}")


def process_folder_for_cropping(folder_path, output_folder, canvas_width=256, canvas_height=256):
    """
    Processes all images in a specified folder, automatically cropping and centering
    each around its region of interest, then resizing and saving to an output folder.
    
    :param folder_path: Path to the folder containing images.
    :param output_folder: Path to the folder where processed images will be saved.
    :param canvas_width: Width of the canvas for the output images.
    :param canvas_height: Height of the canvas for the output images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, f"{filename}")
            auto_crop_and_center_image(image_path, output_path, canvas_width, canvas_height)
            print(f"Processed {filename}")


def check_cropping_quality(image_path):
    """
    Checks if an image has been badly cropped by determining if there are straight lines from bad cropping.

    :param image_path: The path to the image to check.
    :return: A boolean indicating if the image is badly cropped.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not open or find the image {image_path}")
        return True 

    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) == 0:
        print(f"No straight lines found in image {image_path}. Assuming bad cropping.")
        return True

    return False

def check_folder_cropping_quality(folder_path):
    """
    Checks the cropping quality of all images in a folder and prints the results.
    
    :param folder_path: Path to the folder containing images.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            if check_cropping_quality(image_path):
                print(f"Image {filename} has been badly cropped.")

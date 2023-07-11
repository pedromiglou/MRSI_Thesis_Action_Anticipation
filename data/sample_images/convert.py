import cv2

def delete_top(image_path, delete_ratio):
    # Load the image
    image = cv2.imread(image_path)

    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate the number of rows to delete based on the delete ratio
    delete_rows = int(height * delete_ratio)
    delete_columns = width - int(width * delete_ratio)

    # Delete the top rows from the image
    image = image[delete_rows:, :delete_columns]

    return image

# Path to the image
image_path = "dataset_preprocessing2_2.png"

# Ratio of the image to delete from the top (0.0 - 1.0)
delete_ratio = 0.15

# Delete the top part of the image
result_image = delete_top(image_path, delete_ratio)

# Overwrite the initial image file
cv2.imwrite(image_path, result_image)

print("Image overwritten successfully.")
import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread('./images/sample.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Unable to load image.")
else:
    # Apply binary thresholding to create a binary image
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    if binary_image is None:
        print("Error: Thresholding failed.")
    else:
        # Find the coordinates of the non-zero pixels
        y_coords, x_coords = np.where(binary_image > 0)

        # Create a matrix of coordinates
        coords_matrix = np.vstack((x_coords, y_coords))

        # Find the coordinates of the non-zero pixels
        y_coords, x_coords = np.where(binary_image > 0)

        # Create a matrix of coordinates
        coords_matrix = np.vstack((x_coords, y_coords))

        # Calculate the mean of the coordinates
        mean_coords = np.mean(coords_matrix, axis=1)

        # Center the coordinates
        centered_coords = coords_matrix - mean_coords[:, np.newaxis]

        # Perform PCA
        cov_matrix = np.cov(centered_coords)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors_sorted = eigenvectors[:, sorted_indices]

        # Calculate the angle of the principal component with the greatest variance
        angle_radians = np.arctan2(eigenvectors_sorted[1, 0], eigenvectors_sorted[0, 0])
        angle_degrees = np.degrees(angle_radians)

        print("Detected angle:", angle_degrees)

        # Draw the major axis line on the image
        center_x, center_y = mean_coords
        line_length = 100  # Adjust the length of the line as needed
        end_x = int(center_x + line_length * np.cos(angle_radians))
        end_y = int(center_y + line_length * np.sin(angle_radians))
        cv2.line(image, (int(center_x), int(center_y)), (end_x, end_y), (255, 0, 0), 2)

        # Visualize the result
        cv2.imshow('Image with Axis', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

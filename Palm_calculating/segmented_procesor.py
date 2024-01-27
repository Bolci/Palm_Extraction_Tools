import numpy as np
import cv2
from copy import copy


class SegmentedProcesor:
    """
       A class for processing segmented images. Provides functionalities to find object boundaries,
       assign points to clusters, draw boundaries, and draw circles around objects.
       """
    def __init__(self) -> None:
        pass

    @staticmethod
    def find_object_boundaries(image: np.array) -> dict:
        """
        Finds and returns the boundaries of unique objects (trees) in a given image.

        Parameters:
        image (np.array): A segmented image where each object (tree) is represented by a unique value
        Returns:
        dict: A dictionary where keys are object values, and values are contours of the objects.
        """

        unique_objects = np.unique(image)
        unique_objects = unique_objects[unique_objects != 0]
        object_boundaries = {}

        for obj in unique_objects:
            mask = np.where(image == obj, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            object_boundaries[obj] = contours

        return object_boundaries

    @staticmethod
    def assign_point_to_cluster_number(instance_segmented_img: np.array, points: np.array) -> dict:
        """
        Assigns each point to a cluster number based on its coordinates in the segmented image.

        Parameters:
        instance_segmented_img (np.array): The segmented image array.
        points (list): List of point coordinates.

        Returns:
        dict: Dictionary mapping cluster numbers to point coordinates.
        """
        results = {}

        for single_point_coordinates in points:
            cluster_number = instance_segmented_img[single_point_coordinates[1], single_point_coordinates[0]]
            results[cluster_number] = single_point_coordinates

        return results

    @staticmethod
    def get_colour() -> np.array:
        """
        Generates a random color.

        Returns:
        np.array: An array representing a random color.
        """
        return np.random.randint(0, 255, size=3).tolist()

    @staticmethod
    def convert_image_to_coloured(image: np.array) -> np.array:
        """
        Converts a grayscale or single-channel image to a colored (BGR) image.

        Parameters:
        image (np.array): Input image array.

        Returns:
        np.array: Colored image array.
        """
        if len(image.shape) == 2 or image.shape[2] == 1:
            colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            colored_image = copy(image)
        return colored_image

    @staticmethod
    def draw_and_boundaries(original_image: np.array, boundaries: dict) -> np.array:
        """
        Draws boundaries around each segmented object in the image.

        Parameters:
        original_image (np.array): The original image array.
        boundaries (dict): Dictionary of object boundaries.

        Returns:
        np.array: Image array with drawn boundaries.
        """
        colored_image = SegmentedProcesor.convert_image_to_coloured(original_image)

        for obj, contours in boundaries.items():
            # Random color for each object
            color = SegmentedProcesor.get_colour()
            cv2.drawContours(colored_image, contours, -1, color, 2)

        return cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def draw_circle(instance_segmented_img: np.array, points: np.array) -> np.array:
        """
        Draws circles around each object based on the segmented image and provided points.

        Parameters:
        instance_segmented_img (np.array): The segmented image array.
        points (list): List of point coordinates.

        Returns:
        np.array: Image array with drawn circles.
        """
        zero_img = np.zeros(instance_segmented_img.shape).astype(np.uint8)
        colored_image = cv2.cvtColor(zero_img, cv2.COLOR_GRAY2BGR)

        unique_objects = np.unique(instance_segmented_img)
        unique_objects = unique_objects[unique_objects != 0]

        # points_with_coordinates = self.assign_point_to_cluster_number(instance_segmented_img, points)

        for id_point, unique_object_id in enumerate(unique_objects):
            coordinates = points[id_point]

            area = np.sum(instance_segmented_img == unique_object_id)
            radius = int(np.round(np.sqrt(area / np.pi)))
            colored_image = cv2.circle(colored_image, (coordinates[1], coordinates[0]), radius, (255, 0, 0), 1)

        return colored_image

    @staticmethod
    def colour_segmented(instance_segmented_img: np.array) -> np.array:
        coloured = np.zeros(instance_segmented_img.shape + (3,))

        for id_object in np.unique(instance_segmented_img)[1:]:
            color = SegmentedProcesor.get_colour()
            j,i = np.where(instance_segmented_img == id_object)
            coloured[j,i,:] = color

        return coloured

    def process(self, instance_segmented_img: np.array, points: np.array) -> tuple:
        instance_segmented_img = copy(instance_segmented_img)
        points = copy(points)

        object_boundaries = self.find_object_boundaries(instance_segmented_img)
        img_zero = np.zeros(instance_segmented_img.shape).astype(np.uint8)
        raster_drawn = self.draw_and_boundaries(img_zero, object_boundaries)
        circled_img = self.draw_circle(instance_segmented_img, points)
        coloured_segmented = self.colour_segmented(instance_segmented_img)

        return circled_img, raster_drawn, coloured_segmented

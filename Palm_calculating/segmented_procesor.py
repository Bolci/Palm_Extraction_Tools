import numpy as np
import cv2
from copy import copy


class SegmentedProcesor:
    def __init__(self):
        pass

    def find_object_boundaries(self, image):
        # Find unique indexes (objects) in the image
        unique_objects = np.unique(image)

        # Skip the background if it's indexed as 0
        unique_objects = unique_objects[unique_objects != 0]

        object_boundaries = {}

        for obj in unique_objects:
            # Create a mask for the current object
            mask = np.where(image == obj, 255, 0).astype(np.uint8)

            # Find contours (boundaries) of the object
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # You might have multiple contours if the object has holes
            object_boundaries[obj] = contours

        return object_boundaries

    def assign_point_to_cluster_number(self, instance_segmented_img, points):
        results = {}

        for single_point_coordinates in points:
            cluster_number = instance_segmented_img[single_point_coordinates[1], single_point_coordinates[0]]
            results[cluster_number] = single_point_coordinates

        return results

    def draw_and_boundaries(self, original_image, boundaries):
        # original_image = original_image.astype(np.uint8)
        # Convert to a color image if it's grayscale
        if len(original_image.shape) == 2 or original_image.shape[2] == 1:
            colored_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            colored_image = original_image.copy()

        for obj, contours in boundaries.items():
            # Random color for each object
            color = np.random.randint(0, 255, size=3).tolist()
            cv2.drawContours(colored_image, contours, -1, color, 2)

        return cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)

    def draw_circle(self, instance_segmented_img, points):
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

    def process(self, instance_segmented_img, points):
        instance_segmented_img = copy(instance_segmented_img)
        points = copy(points)

        object_boundaries = self.find_object_boundaries(instance_segmented_img)
        img_zero = np.zeros(instance_segmented_img.shape).astype(np.uint8)
        raster_drawn = self.draw_and_boundaries(img_zero, object_boundaries)

        circled_img = self.draw_circle(instance_segmented_img, points)

        return circled_img, raster_drawn

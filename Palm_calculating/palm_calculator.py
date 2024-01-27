from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter
from mmseg.apis import init_model, inference_model
from mmengine import Config
from copy import copy
import cv2
import numpy as np
import torch


class PalmCalculator:
    def __init__(self):
        self.model = None

    @staticmethod
    def detect_peaks(image: np.array) -> np.array:
        """
        Detects peaks in a given image using a maximum filter and binary erosion for peak detection.

        :param image: Input image where peaks are to be detected.
        :return: A boolean array with True at peaks locations.
        """
        neighborhood = generate_binary_structure(2, 2)
        local_max = maximum_filter(image, footprint=neighborhood) == image
        background = (image == 0)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
        detected_peaks = local_max ^ eroded_background

        return detected_peaks

    def set_model(self, config_path: str, checkpoint_path: str, device: str = 'cpu') -> None:
        """
        Loads a model for palm tree segmentation using provided configuration and checkpoint paths.

        :param config_path: Path to the model's configuration file.
        :param checkpoint_path: Path to the model's checkpoint file.
        :param device: Device to run the model on ('cpu' or 'cuda').
        """
        config_loaded = Config.fromfile(config_path)
        self.model = init_model(config_loaded, checkpoint_path, device=device)

    @torch.no_grad()
    def inference(self, img_loaded: np.array, class_id: int = 1) -> tuple:
        """
        Performs inference on an image using the loaded model.

        :param img_loaded: The loaded image for inference.
        :param class_id: The class ID to be considered for segmentation. Default is 1.
        :return: A tuple containing the class map and logits map.
        """

        result = inference_model(self.model, img_loaded)
        class_map = result.pred_sem_seg.data[0].detach().cpu().numpy()
        logits_map = result.seg_logits.data.detach().cpu().numpy()

        return class_map, logits_map[class_id]

    @staticmethod
    def threshold_logits(logits_map: np.array, threshold: int = 0) -> np.array:
        """
        Applies a threshold to the logits map.

        :param logits_map: The logits map to be thresholded.
        :param threshold: The threshold value. Default is 0.
        :return: The thresholded logits map.
        """
        log_min_max = copy(logits_map)
        log_min_max[logits_map <= threshold] = 0
        return log_min_max

    def get_peaks(self, logits_map: np.array) -> np.array:
        """
        Applies Gaussian blurring and detects peaks in the logits map.

        :param logits_map: The logits map for peak detection.
        :return: A boolean array indicating the peaks in the logits map.
        """
        blured_img = cv2.GaussianBlur(logits_map, (5, 5), 0)
        peaks = self.detect_peaks(blured_img)

        return peaks

    def calculate_palms(self, loaded_img: np.array) -> int:
        """
        Calculates the number of palm clusters in an image.

        :param loaded_img: The loaded image for palm calculation.
        :return: The number of detected palm clusters.
        """
        class_map, logits = self.inference(loaded_img)
        log_min_max = self.threshold_logits(logits)
        peaks = self.get_peaks(log_min_max)

        n_of_clusters = len(np.where(peaks)[0])
        return n_of_clusters

    @staticmethod
    def calculate_distance_from_points(img_coord: np.array, point: np.array) -> np.array:
        """
        Calculates Euclidean distance from each point in the image to a specific point.

        :param img_coord: Coordinates in the image.
        :param point: The specific point to calculate distance to.
        :return: Array of distances.
        """
        new_coord = copy(img_coord)
        new_point = copy(point)
        return np.sqrt(np.sum(np.power(new_coord - new_point, 2), axis=1))

    @staticmethod
    def assign_to_cluster(distances:np.array) -> np.array:
        """
        Assign each point to the nearest cluster based on distances.

        :param distances: Distances of points to cluster centers.
        :return: Array of cluster indices each point is assigned to.
        """
        dist = copy(distances)
        return np.argmin(dist, axis=0)

    @staticmethod
    def draw_points(new_img: np.array, i: np.array, j: np.array) -> np.array:
        """
        Draw points on an image at specified coordinates.

        :param new_img: Image to draw points on.
        :param i: Array of i (row) coordinates.
        :param j: Array of j (column) coordinates.
        :return: Image with points drawn.
        """
        n_of_clusters = len(i)
        for x_cluster in range(n_of_clusters):
            point_cluster = np.asarray([i[x_cluster], j[x_cluster]])
            new_img = cv2.circle(new_img, point_cluster, radius=4, color=(255), thickness=-1)
        return new_img

    def segment_palms(self, loaded_img: np.array, image_path: str) -> tuple:
        """
        Segment palm trees in an image and calculate their coordinates.

        :param loaded_img: Loaded input image.
        :param image_path: Path to the input image.
        :return: Tuple with segmented image, logits, point image, number of clusters, and peaks coordinates.
        """
        class_map, logits = self.inference(loaded_img, image_path)
        log_min_max = self.threshold_logits(logits)
        peaks = self.get_peaks(log_min_max)

        j, i = np.where((class_map == peaks) & (class_map == 1))
        new_peak_map = np.zeros(class_map.shape)
        new_peak_map[j, i] = 1

        n_of_clusters = len(j)

        peaks_coordinates = np.concatenate((j.reshape((1,) + j.shape), i.reshape((1,) + i.shape)), axis=0).T

        img_j, img_i = np.where(class_map == 1)
        img_j = img_j.reshape(1, -1)
        img_i = img_i.reshape(1, -1)
        cat_coord = np.concatenate((img_j, img_i), axis=0).T

        distances = []

        for x_cluster in range(n_of_clusters):
            point_cluster = np.asarray([j[x_cluster], i[x_cluster]])
            point_cluster = point_cluster.reshape((1,) + point_cluster.shape)

            single_distance = self.calculate_distance_from_points(cat_coord, point_cluster)
            my_distance = single_distance.copy()

            distances.append(my_distance)

        distances = np.asarray(distances)
        asigned = self.assign_to_cluster(distances)

        segmented_img = np.zeros(class_map.shape)
        point_img = np.zeros(class_map.shape)

        point_img = self.draw_points(point_img, i, j)

        all_colours = np.arange(1, 3000)
        np.random.shuffle(all_colours)

        for x_coord, coordinates in enumerate(cat_coord):
            colo = all_colours[asigned[x_coord]]
            segmented_img[coordinates[0]][coordinates[1]] = colo

        n_of_clusters = len(np.unique(segmented_img)) - 1

        return segmented_img, logits, point_img, n_of_clusters, peaks_coordinates



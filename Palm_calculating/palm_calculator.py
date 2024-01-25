from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter
from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmengine import Config
from copy import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


class PalmCalculator:
    def __init__(self, ):
        self.model = None

    def detect_peaks(self, image):
        neighborhood = generate_binary_structure(2, 2)
        local_max = maximum_filter(image, footprint=neighborhood) == image
        background = (image == 0)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
        detected_peaks = local_max ^ eroded_background

        return detected_peaks

    def set_model(self, config_path, checkpoint_path, device='cpu'):
        config_loaded = Config.fromfile(config_path)
        self.model = init_model(config_loaded, checkpoint_path, device=device)

    @torch.no_grad()
    def inference(self, img_loaded, class_id=1):
        # img_loaded = torch.tensor(img_loaded)
        result = inference_model(self.model, img_loaded)
        class_map = result.pred_sem_seg.data[0].detach().cpu().numpy()
        logits_map = result.seg_logits.data.detach().cpu().numpy()

        return class_map, logits_map[1]

    def threshold_logits(self, logits_map, threshold=0):
        log_min_max = copy(logits_map)
        log_min_max[logits_map <= threshold] = 0
        return log_min_max

    def get_peaks(self, logits_map):
        blured_img = cv2.GaussianBlur(logits_map, (5, 5), 0)
        blured_img_mask = blured_img.astype(bool)
        peaks = self.detect_peaks(blured_img)

        return peaks

    def calculate_palms(self, loaded_img, image_path):
        class_map, logits = self.inference(loaded_img)
        log_min_max = self.threshold_logits(logits)
        peaks = self.get_peaks(log_min_max)

        n_of_clusters = len(np.where(peaks)[0])
        return n_of_clusters

    def calculate_distance_from_points(self, img_coord, point):
        new_coord = copy(img_coord)
        new_point = copy(point)
        return np.sqrt(np.sum(np.power(new_coord - new_point, 2), axis=1))

    def assign_to_cluster(self, distances):
        dist = copy(distances)
        return np.argmin(dist, axis=0)

    def draw_points(self, new_img, i, j):
        n_of_clusters = len(i)
        for x_cluster in range(n_of_clusters):
            point_cluster = np.asarray([i[x_cluster], j[x_cluster]])
            new_img = cv2.circle(new_img, point_cluster, radius=4, color=(255), thickness=-1)
        return new_img

    def segment_palms(self, loaded_img, image_path):

        class_map, logits = self.inference(loaded_img, image_path)
        log_min_max = self.threshold_logits(logits)
        peaks = self.get_peaks(log_min_max)

        j, i = np.where((class_map == peaks) & (class_map == 1))
        new_peak_map = np.zeros(class_map.shape)
        new_peak_map[j, i] = 1

        plt.figure()
        plt.imshow(new_peak_map)

        n_of_clusters = len(j)
        print(n_of_clusters)
        #j, i = np.where(peaks)

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



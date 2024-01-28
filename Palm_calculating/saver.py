import os
from PIL import Image
import rasterio
import numpy as np
from utils import clean_img_extension, make_folders_if_not_exists


class Saver:
    def __init__(self,
                 saving_path: str,
                 inferenced_folder: str = 'inferenced',
                 geofererenced_folder: str = 'georeferenced') -> None:
        """
        Initialize the Saver class with paths for saving images and georeferenced data.

        :param saving_path: The base path where images will be saved.
        :param inferenced_folder: The folder name for saving inferenced images.
        :param geofererenced_folder: The folder name for saving georeferenced images.
        """
        self.saving_path = saving_path
        self.inferenced_folder = inferenced_folder
        self.geofererenced_folder = geofererenced_folder

        self.destination_folder = self.geofererenced_folder
        self.saving_function = self.save_georeferenced

    @staticmethod
    def save_img(img_to_save: np.array, path: str) -> None:
        """
        Save an image to a specified path.

        :param img_to_save: The numpy array representing the image to be saved.
        :param path: The path where the image will be saved.
        """
        path = path % ('.png',)
        Image.fromarray(img_to_save).save(path)

    @staticmethod
    def save_georeferenced(data: tuple, path: str) -> None:
        """
        Save georeferenced data (image and metadata) to a specified path.

        :param data: A tuple containing the image data and metadata.
        :param path: The path where the georeferenced data will be saved.
        """
        img, meta = data
        path = path % ('.tif', )
        with rasterio.open(path, 'w', **meta) as dest:
            dest.write(img)

    def get_paths(self, image_name: str, n_trees: int, image_types: list[str]) -> list[str]:
        """
        Generate file paths for saving images.

        :param image_name: The base name of the image.
        :param n_trees: The number of trees or clusters detected in the image.
        :param image_types: Types of images to save (e.g., segmented, original).
        :return: A list of paths for saving the images.
        """
        saving_path_full = os.path.join(self.saving_path, self.destination_folder)
        make_folders_if_not_exists(saving_path_full)
        return [os.path.join(saving_path_full,  f"{image_name}_{x}_no_trees={n_trees}%s") for x in image_types]

    def save(self,
             img_name: str,
             n_of_clusters: int,
             data_to_save: dict,
             is_georeferenced: bool = True) -> None:
        """
        Save images based on whether they are georeferenced or not.

        :param img_name: Name of the image file.
        :param n_of_clusters: Number of clusters detected in the image.
        :param data_to_save: Dictionary containing data to be saved.
        :param is_georeferenced: Flag to indicate if the data is georeferenced.
        """

        img_name = clean_img_extension(img_name)

        if not is_georeferenced:
            self.destination_folder = self.inferenced_folder
            self.saving_function = self.save_img
            data_to_save = {k:v[0] for k,v in data_to_save.items()}

        saving_paths = self.get_paths(img_name, n_of_clusters, list(data_to_save.keys()))
        for path_to_save, data_to_save in zip(saving_paths, data_to_save.values()):
            self.saving_function(data_to_save, path_to_save)

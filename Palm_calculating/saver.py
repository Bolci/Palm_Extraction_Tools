import os
from PIL import Image
import rasterio
import numpy as np


class Saver:
    def __init__(self,
                 saving_path: str,
                 inferenced_folder: str = 'inferenced',
                 geofererenced_folder: str = 'georeferenced') -> None:
        self.saving_path = saving_path
        self.inferenced_folder = inferenced_folder
        self.geofererenced_folder = geofererenced_folder

        self.destination_folder = self.geofererenced_folder
        self.saving_function = self.save_georeferenced

    @staticmethod
    def clean_img_extension(img_name) -> str:
        return img_name[:-(len(img_name.split('.')[-1]) + 1)]

    @staticmethod
    def make_folders_if_not_exists(saving_path) -> None:
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)

    @staticmethod
    def save_img(img_to_save: np.array, path: str) -> None:
        path = path % ('.png',)
        Image.fromarray(img_to_save).save(path)

    @staticmethod
    def save_georeferenced(data: tuple, path: str):

        img, meta = data
        path = path % ('.tif', )
        with rasterio.open(path, 'w', **meta) as dest:
            dest.write(img)

    def get_paths(self, image_name: str, n_trees: int, image_types: list[str]) -> list[str]:
        saving_path_full = os.path.join(self.saving_path, self.destination_folder)
        self.make_folders_if_not_exists(saving_path_full)
        return [os.path.join(saving_path_full,  f"{image_name}_{x}_no_trees={n_trees}%s") for x in image_types]

    def save(self,
             img_name: str,
             n_of_clusters: int,
             data_to_save: dict,
             is_georeferenced: bool = True):

        img_name = self.clean_img_extension(img_name)

        if not is_georeferenced:
            self.destination_folder = self.inferenced_folder
            self.saving_function = self.save_img
            data_to_save = {k:v[0] for k,v in data_to_save.items()}

        saving_paths = self.get_paths(img_name, n_of_clusters, list(data_to_save.keys()))
        for path_to_save, data_to_save in zip(saving_paths, data_to_save.values()):
            self.saving_function(data_to_save, path_to_save)

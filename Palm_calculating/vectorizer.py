import rasterio
from shapely.geometry import Polygon, LinearRing, Point
from shapely.affinity import affine_transform
import geopandas as gpd
from skimage.measure import find_contours as fc
import numpy as np
from utils import clean_img_extension, make_folders_if_not_exists
import os


class Vectorizer:
    """
    This class is used for vectorizing predicted segmentation masks
    """

    def __init__(self) -> None:
        self.shape_type = {'Polygon': Polygon, 'LinearRing': LinearRing}

    @staticmethod
    def _load_image( image_name: str) -> tuple:
        """Loads an image using rasterio and retrieves its transform and CRS (Coordinate Reference System).

        Args:
            image_name (str): The file path of the image to load.
        Returns:
            tuple: A tuple containing the image array, transform, and CRS.
        """

        with rasterio.open(image_name) as src:
            image = src.read(1)
            transform = src.transform
            crs = src.crs

        return image, transform, crs

    @staticmethod
    def _apply_transforms(all_cons, transform, shape_type) -> list:
        """Applies affine transformations to contours.

        Args:
            all_cons: The contours to transform.
            transform: The affine transformation to apply.

        Returns:
            list: A list of transformed geometries.
        """

        geoms = [shape_type(c) for c in all_cons]
        geoms_transformed = [
            affine_transform(geom, [transform.b, transform.a, transform.e, transform.d, transform.xoff, transform.yoff])
            for geom in geoms]

        return geoms_transformed

    @staticmethod
    def get_geopandas(crs, input_geom) -> gpd.GeoDataFrame:
        """Creates a GeoDataFrame from geometries and a specified CRS.

        Args:
            crs: The Coordinate Reference System to use.
            input_geom: The geometries to include in the GeoDataFrame.

        Returns:
            GeoDataFrame: A GeoDataFrame containing the specified geometries and CRS.
        """

        return gpd.GeoDataFrame(crs=crs, geometry=input_geom)

    @staticmethod
    def _save_polygons(gdf: gpd.GeoDataFrame, output_file: str) -> None:
        """Saves a GeoDataFrame of polygons to a GeoJSON file.

        Args:
            gdf (GeoDataFrame): The GeoDataFrame containing polygons.
            output_file (str): The file path where the GeoJSON will be saved.
        """
        gdf.to_file(output_file, driver='GeoJSON')

    @staticmethod
    def _indentify_inner_polygons( gdf: gpd.GeoDataFrame) -> tuple:
        """Identifies inner polygons within a GeoDataFrame.

        Args:
            gdf (GeoDataFrame): The GeoDataFrame to analyze.

        Returns:
            tuple: A tuple containing two GeoDataFrames - one for outer polygons and one for inner polygons.
        """

        gdf_inner_polygons = gpd.GeoDataFrame().reindex_like(gdf)

        for index, polygon in gdf.iterrows():
            is_within_other = gdf.geometry.apply(
                lambda x: polygon.geometry.within(x) if x != polygon.geometry else False)

            if is_within_other.any():
                row_to_move = gdf.loc[index]
                gdf = gdf.drop(index)
                gdf_inner_polygons.loc[len(gdf_inner_polygons.index)] = row_to_move

        return gdf, gdf_inner_polygons

    def clean_inner_polygons(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Removes inner polygons from a GeoDataFrame by performing a geometric difference operation.

        Args:
            gdf (GeoDataFrame): The GeoDataFrame to clean.

        Returns:
            GeoDataFrame: A cleaned GeoDataFrame with inner polygons removed.
        """
        gdf, innver_poly = self._indentify_inner_polygons(gdf)

        combined_geometry = gdf.unary_union
        combined_geometry_invert = innver_poly.unary_union

        combined_geometry = combined_geometry.difference(combined_geometry_invert)
        gdf = gpd.GeoDataFrame(geometry=[poly for poly in combined_geometry.geoms], crs=gdf.crs)

        return gdf

    def get_polygons(self,
                     image_name: str,
                     predicted_mask: np.array,
                     threshold: float = 0.5,
                     shape_type: str = 'Polygon'):
        """Converts an image to polygons, cleans the polygons, and saves them to a GeoJSON file.

        Args:
            image_name (str): The file path of the image to convert.
            predicted_mask (np.array): predicted mask from segmentation model
            threshold (float, optional): The threshold to use when identifying contours. Defaults to 0.5.
            shape_type (str): The type of shape to create. Defaults to 'Polygon'.
        """

        image, transform, crs = self._load_image(image_name)

        all_ids = np.unique(predicted_mask)
        all_ids = all_ids[all_ids != 0]

        all_polygons = []

        for id_object in all_ids:
            binary_mask = predicted_mask == id_object
            all_cons = fc(binary_mask, threshold)
            found_polygons = self._apply_transforms(all_cons, transform, self.shape_type[shape_type])
            all_polygons += found_polygons
        gdf = self.get_geopandas(crs, all_polygons)
        print('created geopandas')
        return gdf

    def get_and_safe_polygons(self,
                              image_path: str,
                              image_name: str,
                              predicted_mask: np.array,
                              output_path: str = './',
                              threshold: float = 0.5,
                              shape_type: str = 'Polygon') -> None:

        """
        Generates and saves polygons from a predicted mask into a GeoJSON file.

        Parameters:
        image_path (str): The path to the input image file.
        image_name (str): The name of the input image file.
        predicted_mask (np.array): The predicted mask array from the segmentation model.
        output_path (str): The base directory where the GeoJSON file will be saved. Defaults to './'.
        threshold (float): The threshold value to determine polygon boundaries. Defaults to 0.5.
        shape_type (str): The type of shape to create. Defaults to 'Polygon'.
        """

        file_name = clean_img_extension(image_name)
        saving_folder_path = os.path.join(output_path, 'Geopandas_files')
        make_folders_if_not_exists(saving_folder_path)
        output_path_full = os.path.join(saving_folder_path,  f'{file_name}_polygons.geojson')

        gdf_file = self.get_polygons(image_path, predicted_mask, threshold, shape_type)

        self._save_polygons(gdf_file, output_path_full)

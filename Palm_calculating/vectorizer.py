import rasterio
from shapely.geometry import Polygon, LinearRing, Point
from shapely.affinity import affine_transform
import geopandas as gpd
from skimage.measure import find_contours as fc
import numpy as np

class Vectorizer:
    """
    This class is used for vectorizing predicted segmentation masks
    """

    def __init__(self):
        self.shape_type = {'Polygon': Polygon, 'LinearRing': LinearRing}

    def _load_image(self, image_name: str) -> tuple:
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

    def _apply_transforms(self, all_cons, transform, shape_type) -> list:
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

    def get_geopandas(self, crs, input_geom):
        """Creates a GeoDataFrame from geometries and a specified CRS.

        Args:
            crs: The Coordinate Reference System to use.
            input_geom: The geometries to include in the GeoDataFrame.

        Returns:
            GeoDataFrame: A GeoDataFrame containing the specified geometries and CRS.
        """

        return gpd.GeoDataFrame(crs=crs, geometry=input_geom)

    def _save_polygons(self, gdf, output_file: str):
        """Saves a GeoDataFrame of polygons to a GeoJSON file.

        Args:
            gdf (GeoDataFrame): The GeoDataFrame containing polygons.
            output_file (str): The file path where the GeoJSON will be saved.
        """
        gdf.to_file(output_file, driver='GeoJSON')

    def _indentify_inner_polygons(self, gdf):
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

    def clean_inner_polygons(self, gdf):
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

    def get_polygons(self, image_name, predicted_mask, output_path='output.geojson', threshold: float = 0.5,
                     shape_type='Polygon'):
        """Converts an image to polygons, cleans the polygons, and saves them to a GeoJSON file.

        Args:
            image_name (str): The file path of the image to convert.
            output_path (str, optional): The file path where the GeoJSON will be saved. Defaults to 'output.geojson'.
            threshold (float, optional): The threshold to use when identifying contours. Defaults to 0.5.
        """

        image, transform, crs = self._load_image(image_name)

        all_ids = np.unique(predicted_mask)
        all_ids = all_ids[all_ids != 0]

        for id_object in all_ids:
            j, i = np.where(predicted_mask == id_object)
            zero_img = np.zeros(predicted_mask.shape)
            zero_img[j, i] = 1
            all_cons = fc(predicted_mask, threshold)

        geoms_transformed = self._apply_transforms(all_cons, transform, self.shape_type[shape_type])
        gdf = self.get_geopandas(crs, geoms_transformed)
        print('created geopandas')
        self._save_polygons(gdf, output_path)
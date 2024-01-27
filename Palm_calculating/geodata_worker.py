import numpy as np
import rasterio
from copy import copy


class GeodataWorker:
    @staticmethod
    def get_rsi_image(image_path: str) -> tuple:
        """
        Reads a remote sensing image (RSI) using rasterio and returns the transposed data array and metadata.

        :param image_path: Path to the RSI file.
        :return: A tuple containing the transposed image data and metadata.
        """
        with rasterio.open(image_path) as src:
            data = src.read().astype(np.float32)
            meta = src.meta

        return data.transpose((1, 2, 0)), meta

    @staticmethod
    def corect_data_to_geotiff(input_data: dict) -> dict:
        """
        Corrects and formats data into a structure suitable for GeoTIFF writing.

        :param input_data: A dictionary containing image data and metadata.
        :return: A dictionary with corrected and formatted data and metadata.
        """

        corected_data = {}
        for key, val in input_data.items():
            data, meta = val

            edit_data = copy(data)
            edit_meta = copy(meta)

            if len(data.shape) == 2:
                edit_data = edit_data.reshape(data.shape + (1,))

            edit_data = edit_data.transpose(2,0,1)
            edit_meta['count'] = edit_data.shape[0]
            corected_data[key] = [edit_data, edit_meta]

        return corected_data

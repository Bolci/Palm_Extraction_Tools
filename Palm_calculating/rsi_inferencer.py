import rasterio
import numpy as np


class RSI_inferencer:
    def write_geotiff(self, out, raster_file, output_file):
        """Writes a georeferenced TIFF file using the output from the segmentation model.

        Args:
            out (ndarray): The output array from the segmentation model.
            raster_file (str): The file path of the original raster file.
            output_file (str): The file path where the georeferenced TIFF will be saved.

        """

        with rasterio.open(raster_file) as src:
            ras_meta = src.profile
            ras_meta.update(count=1)
            with rasterio.open(output_file, 'w', **ras_meta) as dst:
                dst.write(np.reshape(out, (1,) + out.shape))

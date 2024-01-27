import os
import numpy as np
from palm_calculator import PalmCalculator
from geodata_worker import GeodataWorker
from segmented_procesor import SegmentedProcesor
from arg_parser import parse_args
from saver import Saver


def main():
    args = parse_args()
    path_img = args.input_folder
    config_path = args.config
    checkpoint_path = args.checkpoint
    output_path = args.out
    is_georeferenced = args.georeference

    all_files_img = os.listdir(path_img)
    all_files_img = [x for x in all_files_img if 'tif' in x and 'tif.aux' not in x]
    all_files_img.sort()

    segmenter_processor = SegmentedProcesor()
    palm_calculator = PalmCalculator()
    palm_calculator.set_model(config_path, checkpoint_path)

    for id_img, img in enumerate(all_files_img):
        print(f'--- processing image {img}---')

        img_path = os.path.join(path_img, img)
        loaded_rsi_img, meta = GeodataWorker.get_rsi_image(img_path)

        (segmented_img, logits,
         point_img, n_of_clusters,
         peaks_coordinates) = palm_calculator.segment_palms(loaded_rsi_img)

        print(f'--- processing number of trees is {n_of_clusters}---')

        class_map, _ = palm_calculator.inference(loaded_rsi_img)
        circle_img, raster_drawn, coloured_segmented = segmenter_processor.process(segmented_img, peaks_coordinates)

        point_img = point_img.astype(np.uint8)
        coloured_segmented = coloured_segmented.astype(np.uint8)

        data_to_save = {'Circles': (circle_img, meta),
                        'raster_drawn': (raster_drawn, meta),
                        'points': (point_img, meta),
                        'segmented_img': (coloured_segmented, meta)}

        if is_georeferenced:
            data_to_save = GeodataWorker.corect_data_to_geotiff(data_to_save)

        img_saver = Saver(output_path)
        img_saver.save(img, n_of_clusters, data_to_save, is_georeferenced=is_georeferenced)


if __name__ == '__main__':
    main()

import os
import numpy as np
from PIL import Image

from palm_calculator import PalmCalculator
from rsi_inferencer import RSI_inferencer
from segmented_procesor import SegmentedProcesor
from vectorizer import Vectorizer
from utils import get_rsi_image, get_printable_rsi, colour_img_new
from arg_parser import parse_args
import matplotlib.pyplot as plt

def main():
    #args = parse_args()

    path_img = '/media/bolci/Extreme SSD/FujirahProject/Palm_tree_segmentation/palm_calculating/Enhanced'
    multi_config_path = "/media/bolci/Extreme SSD/FujirahProject/Palm_tree_segmentation/palm_checkpoints/configs_multi/mask2former.py"
    multi_checkpoint_path = '/media/bolci/Extreme SSD/FujirahProject/Palm_tree_segmentation/palm_checkpoints/checkpoints_multi/mask2former.pth'

    all_files_img = os.listdir(path_img)
    all_files_img = [x for x in all_files_img if 'tif' in x and 'tif.aux' not in x]
    all_files_img.sort()

    checkpoint_path = multi_checkpoint_path
    config_path = multi_config_path

    segmenter_processor = SegmentedProcesor()
    palm_calculator = PalmCalculator()
    palm_calculator.set_model(config_path, checkpoint_path)

    for id_img, img in enumerate(all_files_img):
        img_path = os.path.join(path_img, img)
        loaded_rsi_img = get_rsi_image(img_path)

        (segmented_img, logits,
         point_img, n_of_clusters,
         peaks_coordinates) = palm_calculator.segment_palms(loaded_rsi_img, img_path)

        class_map, _ = palm_calculator.inference(loaded_rsi_img, img_path)

        pritable_rsi = get_printable_rsi(loaded_rsi_img)

        circle_img, raster_drawn = segmenter_processor.process(segmented_img, peaks_coordinates)

        plt.figure()
        plt.imshow(circle_img)
        plt.show()

        break

        # inferencer_rsi.write_geotiff(gray_mask, img_path, f'inferenced_img/inferenced_{id_img}.tif')

        # Image.fromarray(gray_mask).save(f'inferenced_img/pritable_rsi_{id_img}_{img}.png')
        # circle_img_gray = cv2.cvtColor(circle_img, cv2.COLOR_RGB2GRAY)
        # Image.fromarray(gray_mask).save(f'inferenced_img/to_predict_{id_img}.png')

        # circle_img = colour_img_new(circle_img_gray)
        Image.fromarray(circle_img).save(f'inferenced_img/circle_{img}_{id_img}.png')
        # Image.fromarray(logits.astype(np.uint8)).save(f'inferenced_img/logits_{id_img}.png')
        # Image.fromarray(rgb_mask.astype(np.uint8)).save(f'inferenced_img/rgb_mask_{id_img}.png')
        Image.fromarray((class_map * 255).astype(np.uint8)).save(f'inferenced_img/class_map_{img}_{id_img}.png')
        segmented_img = colour_img_new(segmented_img)
        Image.fromarray(segmented_img.astype(np.uint8)).save(
            f'inferenced_img/segmented_img_{img}_no_palms_{n_of_clusters}.png')
        Image.fromarray(point_img.astype(np.uint8)).save(f'inferenced_img/point_img_{img}_{id_img}.png')
        # Image.fromarray(raster_drawn.astype(np.uint8)).save(f'inferenced_img/raster_drawn_{id_img}.png')
        # vectorizer.get_polygons(img_path, circle_img_gray, shape_type='LinearRing')
        Image.fromarray(raster_drawn.astype(np.uint8)).save(f'inferenced_img/raster_drawn_{img}_{id_img}.png')

        '''
        plt.figure()
        plt.imshow(circle_img)
        Image.fromarray(circle_img).save('circle_img.png')
        #plt.imsave('prediction.png',pritable_rsi,  dpi=600)

        plt.figure()
        plt.imshow(masked_printable_img)
        Image.fromarray(masked_printable_img).save(f'masked_printable_img.png')
        #plt.imsave( 'ground_truth.png',masked_printable_img, dpi=600)
        '''
        # plt.figure()
        # plt.imshow(logits)
        # Image.fromarray(logits.astye).save('logits.png')
        # plt.imsave(f'inferenced_img/logits_{id_img}.png', logits, dpi=600)

        # plt.figure()
        # plt.imshow(rgb_mask)

        # plt.figure()
        # plt.imshow(point_img)

        # print(segmented_img.shape)
        # plt.figure()
        # plt.imshow(segmented_img)

        # plt.figure()
        # plt.imshow(circle_img)

        # plt.figure()
        # plt.imshow(rgb_mask)
        # plt.imshow(point_img, alpha=0.5)

        # plt.figure()
        # plt.imshow(class_map)
        # plt.imshow(point_img, alpha=0.5)

        # plt.figure()
        # plt.imshow(rgb_mask)
        # plt.imshow(class_map, alpha=0.5)

        # plt.figure()
        # plt.imshow(class_map)

        # plt.show()
        # print(f'number of palms is {n_of_clusters}')

        print(f'image {img} palm is {n_of_clusters}')

        '''
        plt.figure()
        plt.imshow(rgb_mask)
        Image.fromarray(rgb_mask.astype(np.uint8)).save('rgb_mask.png')
        #plt.imsave( 'rgb_mask.png',rgb_mask, dpi=600)

        plt.figure()
        plt.imshow(class_map, cmap='gray')
        Image.fromarray((class_map*255).astype(np.uint8)).save('class_map.png')
        #plt.imsave('class_mal.png',class_map,  dpi=600)

        plt.figure()
        plt.imshow(colour_img_new(segmented_img))
        Image.fromarray(segmented_img.astype(np.uint8)).save('segmented_img.png')
        #plt.imsave( 'segmented.png', segmented_img, dpi=600)

        plt.figure()
        plt.imshow(point_img, cmap='gray')
        Image.fromarray(point_img.astype(np.uint8)).save('point_img.png')
        #plt.imsave('points.png', point_img, dpi=600)

        plt.figure()
        plt.imshow(raster_drawn, cmap='gray')
        Image.fromarray(raster_drawn.astype(np.uint8)).save('raster_drawn.png')
        #plt.imsave('points.png', point_img, dpi=600)
        '''

if __name__ == '__main__':
    main()




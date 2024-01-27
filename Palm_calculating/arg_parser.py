import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Split and calculate separated palms')
    parser.add_argument('input_folder', help='path folder with input images')
    parser.add_argument('config', help='path to configuration to mmsegmentation model')
    parser.add_argument('checkpoint', help='path to checkpoint to mmsegmentation model')
    parser.add_argument(
        '--out',
        action='store_true',
        default='./',
        help='output path to save precited images')
    parser.add_argument('--georeference', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args
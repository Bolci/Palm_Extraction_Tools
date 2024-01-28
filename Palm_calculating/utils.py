import os


def clean_img_extension(img_name: str) -> str:
    """
    Remove the file extension from an image name.

    :param img_name: The name of the image file.
    :return: Image name without the extension.
    """
    return img_name[:-(len(img_name.split('.')[-1]) + 1)]


def make_folders_if_not_exists(folder_path: str) -> None:
    """
    Create a folder if it does not exist.

    :param folder_path: The path where the folder should be created.
    """
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

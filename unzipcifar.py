import os
import zipfile


def unzip_cifar():
    path_to_zip_file = os.path.join(
        'drive/MyDrive/dataset/', "state_dicts.zip")
    directory_to_extract_to = os.path.join('/content/git-modules/', "cifar10_models")
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
        print("Unzip file successful!")

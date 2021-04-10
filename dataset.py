class Dataset:
    # rel_path = ''
    # drive_path = 'drive/MyDrive/dataset/'
    # image_path = ''

    def __init__(self, image_path: str, classes: []):
        # self.image_path = image_path
        self.classes = classes
        self.class_num = len(classes)
        self.path = 'drive/MyDrive/dataset/' + image_path


# region uc_landuse
uc_classes = ['airplane',
              'chaparral',
              'forest',
              'buildings',
              'agricultural',
              'baseballdiamond',
              'denseresidential',
              'beach',
              'freeway',
              'parkinglot',
              'harbor',
              'golfcourse',
              'intersection',
              'overpass',
              'mobilehomepark',
              'mediumresidential',
              'river',
              'runway',
              'storagetanks',
              'sparseresidential',
              'tenniscourt']
uc_classes.sort()
uc_landuse_ds = Dataset('UCMerced_LandUse/Images', uc_classes)
# endregion

# region big_earth

# endregion

# region resic45

# endregion

# region euro_sat

# endregion

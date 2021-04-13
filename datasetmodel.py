class DatasetModel:
    def __init__(self, name: str, images_path: str, classes: []):
        self.name = name
        self.classes = classes
        self.class_num = len(classes)
        self.path = 'drive/MyDrive/dataset/' + images_path


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
uc_landuse_ds = DatasetModel(
    'UCMerced_LandUse', 'UCMerced_LandUse/Images', uc_classes)
# endregion

# region big_earth

# endregion

# region resic45

# endregion

# region euro_sat

# endregion

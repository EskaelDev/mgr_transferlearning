from typing import List


class DatasetModel:
    def __init__(self, name: str, images_path: str, classes: List[int]):
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
resisc_classes = ["airplane",
                 "airport",
                 "baseball_diamond",
                 "basketball_court",
                 "beach",
                 "bridge",
                 "chaparral",
                 "church",
                 "circular_farmland",
                 "cloud",
                 "commercial_area",
                 "dense_residential",
                 "desert",
                 "forest",
                 "freeway",
                 "golf_course",
                 "ground_track_field",
                 "harbor",
                 "industrial_area",
                 "intersection",
                 "island",
                 "lake",
                 "meadow",
                 "medium_residential",
                 "mobile_home_park",
                 "mountain",
                 "overpass",
                 "palace",
                 "parking_lot",
                 "railway",
                 "railway_station",
                 "rectangular_farmland",
                 "river",
                 "roundabout",
                 "runway",
                 "sea_ice",
                 "ship",
                 "snowberg",
                 "sparse_residential",
                 "stadium",
                 "storage_tank",
                 "tennis_court",
                 "terrace",
                 "thermal_power_station",
                 "wetland"]
# endregion
resisc_classes.sort()
resisc_ds = DatasetModel(
    'RESISC45', 'RESISC45/Images', resisc_classes)
# region euro_sat

# endregion

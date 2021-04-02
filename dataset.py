class Dataset:
    rel_path = ''
    root_path = 'drive/MyDrive/dataset/'
    image_path = ''

    def __init__(self, image_path: str, classes: set):
        self.image_path = image_path
        self.classes = classes
        self.class_num = len(classes)


uc_landuse = Dataset('UCMerced_LandUse/Images')

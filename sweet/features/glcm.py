from skimage.feature import greycomatrix
from skimage.feature import greycoprops

class GLCM(object):
    def __init__(self, image_1, image_2, patches=None):
        """
        Initialise the GLCM object.

        These resources are helpful for understanding GLCMs.
            - http://www.gruppofrattura.it/pdf/rivista/numero16/numero_16_art_5.pdf

        Args:
            image_1: An opened OpenCV image file (cv2.imread...)
            image_2: An opened OpenCV image file (cv2.imread...)
            angles:  A list containing angles, e.g. [0, 45, 90, 135]
            dist:    A list containing pixel distances, e.g. [0, 1, 2, 3]
        """
        self.image_1 = image_1
        self.image_2 = image_2

        self.angles = angles
        self.dists = dists

        self.images = []
        self.images.append(self.image_1)
        self.images.append(self.image_2)


    def extract_features(self):
        #xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        #ys.append(greycoprops(glcm, 'correlation')[0, 0])
        glcms = []
        for image in self.images:
            glcm = greycomatrix(patch, angles, dists, 256, symmetric=True, normed=True)
            glcms.append(glcm)
        return glcms


    def squared_error(self):
        glcms = self.get_features()
        raise Exception("Not implemented yet")

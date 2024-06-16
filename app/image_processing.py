# image_processing.py

from hallo.datasets.image_processor import ImageProcessor

class ImageProcessing:
    def __init__(self, config):
        self.config = config
    
    def preprocess(self, source_image_path, save_path):
        img_size = (self.config.data.source_image.width, self.config.data.source_image.height)
        with ImageProcessor(img_size, self.config.face_analysis.model_path) as image_processor:
            source_image_pixels, \
            source_image_face_region, \
            source_image_face_emb, \
            source_image_full_mask, \
            source_image_face_mask, \
            source_image_lip_mask = image_processor.preprocess(
                source_image_path, save_path, self.config.face_expand_ratio)
        return source_image_pixels, source_image_face_region, source_image_face_emb, \
               source_image_full_mask, source_image_face_mask, source_image_lip_mask

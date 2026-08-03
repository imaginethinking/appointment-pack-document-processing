import cv2
import numpy as np
from PIL import Image


class ImagePreprocessingService:
    ADAPTIVE_THRESHOLD_BLOCK_SIZE = 31
    ADAPTIVE_THRESHOLD_CONSTANT = 15

    def __init__(
        self,
        enabled: bool,
        minimum_image_width: int,
    ) -> None:
        self.enabled = enabled
        self.minimum_image_width = minimum_image_width

    def preprocess(
        self,
        image: Image.Image,
    ) -> Image.Image:
        rgb_image = image.convert("RGB")

        if not self.enabled:
            return rgb_image

        image_array = np.asarray(rgb_image)

        grayscale_image = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY,
        )

        resized_image = self._resize_if_required(
            grayscale_image
        )

        thresholded_image = cv2.adaptiveThreshold(
            resized_image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=self.ADAPTIVE_THRESHOLD_BLOCK_SIZE,
            C=self.ADAPTIVE_THRESHOLD_CONSTANT,
        )

        return Image.fromarray(thresholded_image)

    def _resize_if_required(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        current_height, current_width = image.shape

        if current_width >= self.minimum_image_width:
            return image

        scale_factor = (
            self.minimum_image_width / current_width
        )

        resized_height = max(
            1,
            round(current_height * scale_factor),
        )

        return cv2.resize(
            image,
            (
                self.minimum_image_width,
                resized_height,
            ),
            interpolation=cv2.INTER_CUBIC,
        )
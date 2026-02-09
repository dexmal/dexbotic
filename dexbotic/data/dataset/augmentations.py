from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.type_definitions import Targets
from albumentations.core.transforms_interface import BaseTransformInitSchema


class PixelAug:
    def __init__(self, policy='v1', p=0.5):
        self.transform = NAME2AUG[policy]()

    def __call__(self, image):
        np_image = np.array(image)
        np_image = self.transform(image=np_image)['image']
        pil_image = Image.fromarray(np_image, mode="RGB")
        return pil_image



class PadToSquare(A.DualTransform):
    """Pad the input image to make it square.

    This transform pads the input image to ensure that its height and width are equal, resulting in a square image.
    The padding is applied equally on all sides of the image, and the padding value can be specified.

    Args:
        fill (tuple[float, ...] | float): Padding value if border_mode is cv2.BORDER_CONSTANT.
        fill_mask (tuple[float, ...] | float): Padding value for mask if border_mode is cv2.BORDER_CONSTANT.
        border_mode (OpenCV flag): OpenCV border mode.
        p (float): Probability of applying the transform. Default: 1.0.
    Targets:
        image, mask, bboxes, keypoints, volume, mask3d
    Image types:
        uint8, float32
    """

    _targets = (Targets.IMAGE, Targets.MASK)

    class InitSchema(BaseTransformInitSchema):
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ]

    def __init__(
        self,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.fill = fill
        self.fill_mask = fill_mask
        self.border_mode = border_mode

    def apply(
        self,
        img: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the PadToSquare transform to an image.

        Args:
            img (np.ndarray): Image to be transformed.
            pad_top (int): Top padding.
            pad_bottom (int): Bottom padding.
            pad_left (int): Left padding.
            pad_right (int): Right padding.
            **params (Any): Additional parameters.

        """
        height, width = img.shape[:2]
        size = max(height, width)
        pad_top = (size - height) // 2
        pad_bottom = size - height - pad_top
        pad_left = (size - width) // 2
        pad_right = size - width - pad_left
        return fgeometric.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.fill,
        )

    def apply_to_mask(
        self,
        mask: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the PadToSquare transform to a mask.

        Args:
            mask (np.ndarray): Mask to be transformed.
            pad_top (int): Top padding.
            pad_bottom (int): Bottom padding.
            pad_left (int): Left padding.
            pad_right (int): Right padding.
            **params (Any): Additional parameters.

        """
        height, width = mask.shape[:2]
        size = max(height, width)
        pad_top = (size - height) // 2
        pad_bottom = size - height - pad_top
        pad_left = (size - width) // 2
        pad_right = size - width - pad_left
        return fgeometric.pad_with_params(
            mask,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.fill_mask,
        )


def policy_v1(p=0.5):
    aug = [
        A.CoarseDropout(num_holes_range=(3, 6), fill='random_uniform',
                        hole_height_range=(0.05, 0.2),
                        hole_width_range=(0.05, 0.2), p=p),
    ]

    return A.Compose(aug)


def policy_v2(p=0.5):
    aug = [
        # A.ColorJitter(p=p),
        # A.Solarize(p=p),
        # A.InvertImg(p=p),
        A.OneOf([A.AdditiveNoise(p=p),
                 A.RGBShift(p=p),
                 A.GaussNoise(p=p, std_range=(0.1, 0.22)),
                 A.SaltAndPepper(p=p),
                 ], p=0.2),
        A.OneOf([A.MotionBlur(p=p),
                 A.MedianBlur(p=p),
                 A.GaussianBlur(p=p),
                 A.Sharpen(p=p),
                 ], p=0.8),

        A.SomeOf([A.RandomGamma(p=p),
                  A.ISONoise(p=0.1),
                  A.Illumination(p=p),
                  A.ShotNoise(p=p, scale_range=(0.02, 0.1))], n=2),

        A.CoarseDropout(num_holes_range=(3, 6), fill='random_uniform',
                        hole_height_range=(0.05, 0.2),
                        hole_width_range=(0.05, 0.2), p=p),
    ]
    return A.SomeOf(aug, n=2)


def policy_v3(p=0.5):
    aug = [
        A.RandomResizedCrop(size=(384, 384), scale=(0.95, 1.0), ratio=(1.0, 1.0), p=p),
        A.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08, p=p),
    ]
    return A.Compose(aug)


def policy_dm0(p=0.5):
    aug = [
        PadToSquare(border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=1.0),
        A.RandomResizedCrop(
            size=(728, 728), scale=(0.95, 0.95), ratio=(1.0, 1.0), p=1.0
        ),
        A.Rotate(limit=(-5, 5), p=1.0),
        A.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.1, p=p),
    ]
    return A.Compose(aug)


def policy_color_dm0(p=0.5):
    aug = [
        PadToSquare(border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=1.0),
        A.Resize(728, 728, p=1.0),
        A.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.1, p=p),
    ]
    return A.Compose(aug)


def policy_pi0(p=0.5):
    aug = [
        PadToSquare(border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=1.0),
        A.RandomResizedCrop(size=(224, 224), scale=(0.95, 0.95), ratio=(1.0, 1.0), p=1.0),
        A.Rotate(limit=(-5, 5), p=1.0),
        A.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.1, p=p),
    ]
    return A.Compose(aug)


def policy_color(p=0.5):
    aug = [
        PadToSquare(border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=1.0),
        A.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.1, p=p),
    ]
    return A.Compose(aug)


def policy_identity(p=0.5):
    return A.Compose([])


NAME2AUG = {
    'v1': policy_v1,
    'v2': policy_v2,
    'v3': policy_v3,
    'pi0': policy_pi0,
    'color': policy_color,
    'identity': policy_identity,
    'dm0': policy_dm0,
    'color_dm0': policy_color_dm0,
}


if __name__ == '__main__':
    import numpy as np
    from PIL import Image

    image = Image.open('test_data/libero_test.png')

    aug = PixelAug(p=1, policy='v3')
    image = aug(image)
    print(type(image))
    image.save('test_data/aug_test.png')

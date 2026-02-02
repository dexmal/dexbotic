from PIL import Image
import torch


class PreprocessRGB:
    def __init__(self, image_processor, image_aspect_ratio=None, augmentations=None, image_pad_mode='mean'):
        self.image_processor = image_processor
        # FIXME: the definition of image_aspect_ratio is confusing, better to be renamed
        self.image_aspect_ratio = image_aspect_ratio
        self.augmentations = augmentations
        self.image_pad_mode = image_pad_mode

    def __call__(self, image) -> torch.Tensor:
        if image is None:
            crop_size = getattr(
                self.image_processor,
                'crop_size',
                self.image_processor.size)
            image = torch.zeros(3, crop_size['height'], crop_size['width'])
        else:
            if self.image_aspect_ratio == 'pad':
                if self.image_pad_mode == 'zero':
                    image = self.expand2square(image, tuple(int(x * 255) for x in [0, 0, 0]))
                else:
                    image = self.expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
            if self.augmentations:
                image = self.augmentations(image=image)
            image = self.image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
        return image

    @staticmethod
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result


class DummyRGBProcessor:

    def __call__(self, image) -> torch.Tensor:
        return torch.zeros(1)

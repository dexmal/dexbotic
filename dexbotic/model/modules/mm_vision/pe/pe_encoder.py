import torch.nn as nn

from transformers import SiglipImageProcessor

from .pe_configuration import get_config


def get_image_processor(image_res: int = 336):
    image_processor = SiglipImageProcessor(
        do_convert_rgb=None,
        do_normalize=True,
        do_rescale=True,
        do_resize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        resample=3,
        rescale_factor=0.00392156862745098,
        size={
            "height": image_res,
            "width": image_res, 
        }
    )

    return image_processor


class PEVisionTower(nn.Module):
    def __init__(self, vision_tower):
        super().__init__()
        self.is_loaded = False
        self.config = get_config(vision_tower)
        self.load_model()

        self.image_processor = get_image_processor(image_res=self.config.image_size)

    def load_model(self):
        if self.is_loaded:
            return

        self.vision_tower = self.config.build_model()
        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower.forward(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                ).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower.forward(
                images.to(device=self.device, dtype=self.dtype)
            ).to(images.dtype)

        return image_features

    @property
    def dtype(self):
        return list(self.vision_tower.parameters())[-1].dtype

    @property
    def device(self):
        return list(self.vision_tower.parameters())[-1].device

    @property
    def hidden_size(self):
        return self.config.width
    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size // 4) ** 2

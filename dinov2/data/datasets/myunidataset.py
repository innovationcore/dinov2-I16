import logging
from typing import Any, Tuple, Callable, Optional
from PIL import Image
from fastai.vision.all import Path, get_image_files, verify_images

from dinov2.data.datasets.extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")
class MyUniDataset(ExtendedVisionDataset):
    def __init__(self, root: str, verify: bool = False, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.root = Path(root).expanduser()
        image_paths = get_image_files(self.root)
        invalid_images = set()
        if verify:
            invalid_images = set(verify_images(image_paths))
        self.image_paths = [p for p in image_paths if p not in invalid_images]

    def get_image_data(self, index: int) -> bytes:
        image_path = self.image_paths[index]
        img = Image.open(image_path)
        num_channels = len(img.getbands())
        width, height = img.size
        #logger.info("0 img type: " + str(type(img)))
        #logger.info("0 Width: " + str(width))
        #logger.info("0 Height: " + str(height))
        #logger.info("0 image mode: " + str(img.mode))
        #logger.info("0 Number of channels: " + str(num_channels))

        return img
        
    def get_target(self, index: int) -> Any:
        return 0

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)

        #width, height = image.size
        #logger.info("2 img type: " + str(type(image)))
        #logger.info("2 Width: " + str(width))
        #logger.info("2 Height: " + str(height))
        #logger.info("2 image mode: " + str(image.mode))

        if self.transforms is not None:
            #logger.info("TRANSFORMS ENABLED")
            image, target = self.transforms(image, target)

        logger.info("3 img global: " + str(image['global_crops']))
        logger.info("3 img type: " + str(type(image)))
        logger.info("3 img keys: " + str(image.keys()))
        logger.info("3 img global: " + str(type(image['global_crops'])))
        #logger.info("3 img : " + str(image))
        exit()

        return image, target
    

from enum import Enum
from typing import Any, Dict, List, Tuple, Callable, Optional
from PIL import Image
from fastai.vision.all import Path, get_image_files, verify_images

from dinov2.data.datasets.extended import ExtendedVisionDataset

class MyDataset(ExtendedVisionDataset):
    def __init__(self, root: str, verify_images: bool = False, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.root = Path(root).expanduser()
        image_paths = get_image_files(self.root)
        invalid_images = set()
        if verify_images:
            invalid_images = set(verify_images(image_paths))
        self.image_paths = [p for p in image_paths if p not in invalid_images]

    def get_image_data(self, index: int) -> bytes:
        image_path = self.image_paths[index]
        img = Image.open(image_path).convert("RGB")
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
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    

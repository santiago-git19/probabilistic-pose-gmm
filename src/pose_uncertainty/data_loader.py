import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any
from pycocotools.coco import COCO
from .utils.types import ImageSample, Keypoint

class PoseDatasetAdapter(ABC):
    """
    Clase abstracta (Interface).
    Obliga a cualquier dataset nuevo a comportarse igual.
    """
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[ImageSample]:
        pass

class COCOLoader(PoseDatasetAdapter):
    def __init__(self, data_root: str, ann_file: str, image_dir: str):
        """
        :param data_root: Ruta base (ej: ./data/coco)
        :param ann_file: Nombre del json (ej: annotations/person_keypoints_val2017.json)
        :param image_dir: Carpeta de imágenes (ej: val2017)
        """
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, image_dir)
        self.ann_path = os.path.join(data_root, ann_file)
        
        # Inicializar API de COCO
        print(f"Cargando anotaciones desde {self.ann_path}...")
        self.coco = COCO(self.ann_path)
        
        # Filtrar solo imágenes con categoría 'person'
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        print(f"Dataset cargado: {len(self.img_ids)} imágenes encontradas.")

    def __len__(self) -> int:
        return len(self.img_ids)

    def _parse_keypoints(self, raw_kps) -> list[Keypoint]:
        # Lógica para convertir formato plano de COCO [x,y,v, x,y,v...] a lista de objetos Keypoint
        # ... (implementación detallada luego)
        return []

    def __iter__(self) -> Iterator[ImageSample]:
        for img_id in self.img_ids:
            # Metadatos de la imagen
            img_info = self.coco.loadImgs(img_id)[0]
            path = os.path.join(self.image_dir, img_info['file_name'])
            
            # Cargar Anotaciones (Bbox y Keypoints)
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            
            # NOTA PARA TFG: 
            # COCO tiene múltiples personas por imagen.
            # Tu pipeline es Single-Person. Aquí seleccionamos la persona más grande (área).
            if not anns: continue
            
            main_person = max(anns, key=lambda x: x['area'])
            bbox = main_person['bbox'] # [x, y, w, h]
            
            # Carga perezosa (Lazy Loading) de la imagen para no saturar RAM
            img_array = cv2.imread(path)
            if img_array is None:
                continue # Skip si la imagen está corrupta
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            yield ImageSample(
                image_id=img_id,
                image_path=path,
                image_array=img_array,
                bbox=tuple(bbox),
                ground_truth_keypoints=self._parse_keypoints(main_person['keypoints']),
                dataset_source="coco"
            )
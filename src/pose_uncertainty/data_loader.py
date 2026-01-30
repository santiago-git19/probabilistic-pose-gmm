import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, List
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

    def _parse_keypoints(self, raw_kps: List[float], num_keypoints: int) -> List[Keypoint]:
        """
        Convert COCO's flat keypoint format to List[Keypoint].
        
        COCO Format: [x1, y1, v1, x2, y2, v2, ..., x17, y17, v17]
        
        Args:
            raw_kps: Flat list of keypoint data (length 51 for COCO: 17 * 3).
            num_keypoints: Number of labeled keypoints (for validation).
        
        Returns:
            List of Keypoint objects with proper naming and confidence.
        
        Confidence Mapping:
            v=0 → confidence=0.0 (not labeled)
            v=1 → confidence=0.5 (occluded, exists but not visible)
            v=2 → confidence=1.0 (visible)
        """
        COCO_KEYPOINT_NAMES = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        keypoints = []
        
        # COCO has 17 keypoints, stored as triplets (x, y, visibility)
        for i in range(17):
            idx = i * 3
            x = raw_kps[idx]
            y = raw_kps[idx + 1]
            v = int(raw_kps[idx + 2])
            
            # Map visibility to confidence
            confidence_map = {0: 0.0, 1: 0.5, 2: 1.0}
            confidence = confidence_map.get(v, 0.0)
            
            keypoints.append(Keypoint(
                id=i,
                x=float(x),
                y=float(y),
                confidence=confidence,
                name=COCO_KEYPOINT_NAMES[i]
            ))
        
        return keypoints

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
                ground_truth_keypoints=self._parse_keypoints(main_person['keypoints'], num_keypoints=17),
                dataset_source="coco"
            )
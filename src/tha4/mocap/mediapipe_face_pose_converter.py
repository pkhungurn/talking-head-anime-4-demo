from abc import ABC, abstractmethod
from typing import List, Callable, Optional

from tha4.mocap.mediapipe_face_pose import MediaPipeFacePose


class MediaPipeFacePoseConverter(ABC):
    @abstractmethod
    def convert(self, mediapipe_face_pose: MediaPipeFacePose) -> List[float]:
        pass

    @abstractmethod
    def init_pose_converter_panel(
            self,
            parent,
            current_pose_supplier: Callable[[], Optional[MediaPipeFacePose]]):
        pass
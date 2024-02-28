import json
import os
from typing import Optional, Dict

import numpy


class MediaPipeFacePose:
    KEY_BLENDSHAPE_PARAMS = "blendshape_params"
    KEY_XFORM_MATRIX = "xform_matrix"

    def __init__(self, blendshape_params: Optional[Dict[str, float]], xform_matrix: Optional[numpy.ndarray]):
        if blendshape_params is None:
            blendshape_params = {}
        if xform_matrix is None:
            self.xform_matrix = numpy.zeros(4, 4)
            for i in range(4):
                self.xform_matrix[i, i] = 1.0

        self.blendshape_params = blendshape_params
        self.xform_matrix = xform_matrix

    def get_json(self):
        return {
            MediaPipeFacePose.KEY_BLENDSHAPE_PARAMS: self.blendshape_params.copy(),
            MediaPipeFacePose.KEY_XFORM_MATRIX: self.xform_matrix.tolist()
        }

    def save(self, file_name: str):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "wt") as fout:
            fout.write(json.dumps(self.get_json()))

    @staticmethod
    def load(file_name: str):
        with open(file_name, "rt") as fin:
            s = fin.read()
            json_data = json.loads(s)
            return MediaPipeFacePose(
                json_data[MediaPipeFacePose.KEY_BLENDSHAPE_PARAMS],
                xform_matrix = numpy.array(json_data[MediaPipeFacePose.KEY_XFORM_MATRIX]))

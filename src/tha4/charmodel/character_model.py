import json
import os.path

import PIL.Image
import torch
from omegaconf import OmegaConf

from tha4.shion.base.image_util import extract_pytorch_image_from_PIL_image
from tha4.poser.modes.mode_14 import create_poser, KEY_FACE_MORPHER, KEY_BODY_MORPHER


class CharacterModel:
    def __init__(self,
                 character_image_file_name: str,
                 face_morpher_file_name: str,
                 body_morpher_file_name: str):
        self.body_morpher_file_name = body_morpher_file_name
        self.face_morpher_file_name = face_morpher_file_name
        self.character_image_file_name = character_image_file_name
        self.poser = None
        self.character_image = None

    def get_poser(self, device: torch.device):
        if self.poser is not None:
            self.poser.to(device)
        else:
            self.poser = create_poser(
                device,
                module_file_names={
                    KEY_FACE_MORPHER: self.face_morpher_file_name,
                    KEY_BODY_MORPHER: self.body_morpher_file_name
                })
        return self.poser

    def get_character_image(self, device: torch.device):
        if self.character_image is None:
            pil_image = PIL.Image.open(self.character_image_file_name)
            if pil_image.mode != 'RGBA':
                raise RuntimeError("Character image is not an RGBA image!")
            self.character_image = extract_pytorch_image_from_PIL_image(pil_image)
        self.character_image = self.character_image.to(device)
        return self.character_image

    def save(self, file_name: str):
        dir = os.path.dirname(file_name)
        rel_char_image_file_name = os.path.relpath(self.character_image_file_name, dir)
        rel_face_morpher_file_name = os.path.relpath(self.face_morpher_file_name, dir)
        rel_body_morpher_file_name = os.path.relpath(self.body_morpher_file_name, dir)
        data = {
            "character_image_file_name": rel_char_image_file_name,
            "face_morpher_file_name": rel_face_morpher_file_name,
            "body_morpher_file_name": rel_body_morpher_file_name,
        }
        conf = OmegaConf.create(data)
        os.makedirs(dir, exist_ok=True)
        with open(file_name, "wt") as fout:
            fout.write(OmegaConf.to_yaml(conf))

    @staticmethod
    def load(file_name: str):
        conf = OmegaConf.to_container(OmegaConf.load(file_name))
        dir = os.path.dirname(file_name)
        character_image_file_name = os.path.join(dir, conf["character_image_file_name"])
        face_morpher_file_name = os.path.join(dir, conf["face_morpher_file_name"])
        body_morpher_file_name = os.path.join(dir, conf["body_morpher_file_name"])
        return CharacterModel(
            character_image_file_name,
            face_morpher_file_name,
            body_morpher_file_name)

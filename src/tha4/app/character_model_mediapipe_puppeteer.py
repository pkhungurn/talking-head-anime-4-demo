import os
import sys
import threading
import time
from typing import Optional
import PIL.Image

import cv2
import mediapipe
from scipy.spatial.transform import Rotation

from tha4.shion.base.image_util import resize_PIL_image
from tha4.charmodel.character_model import CharacterModel
from tha4.image_util import convert_linear_to_srgb
from tha4.mocap.mediapipe_constants import HEAD_ROTATIONS, HEAD_X, HEAD_Y, HEAD_Z
from tha4.mocap.mediapipe_face_pose import MediaPipeFacePose
from tha4.mocap.mediapipe_face_pose_converter_00 import MediaPoseFacePoseConverter00

sys.path.append(os.getcwd())

import torch
import wx


class FpsStatistics:
    def __init__(self):
        self.count = 100
        self.fps = []

    def add_fps(self, fps):
        self.fps.append(fps)
        while len(self.fps) > self.count:
            del self.fps[0]

    def get_average_fps(self):
        if len(self.fps) == 0:
            return 0.0
        else:
            return sum(self.fps) / len(self.fps)


class MainFrame(wx.Frame):
    IMAGE_SIZE = 512

    def __init__(self,
                 pose_converter: MediaPoseFacePoseConverter00,
                 video_capture,
                 face_landmarker,
                 device: torch.device):
        super().__init__(None, wx.ID_ANY, "THA4 Character Model MediaPipe Puppeteer")
        self.face_landmarker = face_landmarker
        self.video_capture = video_capture
        self.pose_converter = pose_converter
        self.device = device

        self.source_image_bitmap = wx.Bitmap(MainFrame.IMAGE_SIZE, MainFrame.IMAGE_SIZE)
        self.result_image_bitmap = wx.Bitmap(MainFrame.IMAGE_SIZE, MainFrame.IMAGE_SIZE)
        self.webcam_capture_bitmap = wx.Bitmap(256, 192)
        self.wx_source_image = None
        self.torch_source_image = None
        self.last_pose = None
        self.mediapipe_face_pose = None
        self.fps_statistics = FpsStatistics()
        self.last_update_time = None
        self.character_model = None
        self.poser = None

        self.create_ui()
        self.create_timers()
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.update_source_image_bitmap()
        self.update_result_image_bitmap()

    def create_timers(self):
        self.capture_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_capture_panel, id=self.capture_timer.GetId())
        self.animation_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_result_image_bitmap, id=self.animation_timer.GetId())

    def on_close(self, event: wx.Event):
        # Stop the timers
        self.animation_timer.Stop()
        self.capture_timer.Stop()

        # Destroy the windows
        self.Destroy()
        event.Skip()

    def on_erase_background(self, event: wx.Event):
        pass

    def create_animation_panel(self, parent):
        self.animation_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.animation_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.animation_panel.SetSizer(self.animation_panel_sizer)
        self.animation_panel.SetAutoLayout(1)

        image_size = MainFrame.IMAGE_SIZE

        if True:
            self.input_panel = wx.Panel(self.animation_panel, size=(image_size, image_size + 128),
                                        style=wx.SIMPLE_BORDER)
            self.input_panel_sizer = wx.BoxSizer(wx.VERTICAL)
            self.input_panel.SetSizer(self.input_panel_sizer)
            self.input_panel.SetAutoLayout(1)
            self.animation_panel_sizer.Add(self.input_panel, 0, wx.FIXED_MINSIZE)

            self.source_image_panel = wx.Panel(self.input_panel, size=(image_size, image_size), style=wx.SIMPLE_BORDER)
            self.source_image_panel.Bind(wx.EVT_PAINT, self.paint_source_image_panel)
            self.source_image_panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
            self.input_panel_sizer.Add(self.source_image_panel, 0, wx.FIXED_MINSIZE)

            self.load_model_button = wx.Button(self.input_panel, wx.ID_ANY, "Load Model")
            self.input_panel_sizer.Add(self.load_model_button, 1, wx.EXPAND)
            self.load_model_button.Bind(wx.EVT_BUTTON, self.load_model)

            self.input_panel_sizer.Fit(self.input_panel)

        if True:
            def current_pose_supplier() -> Optional[MediaPipeFacePose]:
                return self.mediapipe_face_pose

            self.pose_converter.init_pose_converter_panel(self.animation_panel, current_pose_supplier)

        if True:
            self.animation_left_panel = wx.Panel(self.animation_panel, style=wx.SIMPLE_BORDER)
            self.animation_left_panel_sizer = wx.BoxSizer(wx.VERTICAL)
            self.animation_left_panel.SetSizer(self.animation_left_panel_sizer)
            self.animation_left_panel.SetAutoLayout(1)
            self.animation_panel_sizer.Add(self.animation_left_panel, 0, wx.EXPAND)

            self.result_image_panel = wx.Panel(self.animation_left_panel, size=(image_size, image_size),
                                               style=wx.SIMPLE_BORDER)
            self.result_image_panel.Bind(wx.EVT_PAINT, self.paint_result_image_panel)
            self.result_image_panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
            self.animation_left_panel_sizer.Add(self.result_image_panel, 0, wx.FIXED_MINSIZE)

            separator = wx.StaticLine(self.animation_left_panel, -1, size=(256, 5))
            self.animation_left_panel_sizer.Add(separator, 0, wx.EXPAND)

            background_text = wx.StaticText(self.animation_left_panel, label="--- Background ---",
                                            style=wx.ALIGN_CENTER)
            self.animation_left_panel_sizer.Add(background_text, 0, wx.EXPAND)

            self.output_background_choice = wx.Choice(
                self.animation_left_panel,
                choices=[
                    "TRANSPARENT",
                    "GREEN",
                    "BLUE",
                    "BLACK",
                    "WHITE"
                ])
            self.output_background_choice.SetSelection(0)
            self.animation_left_panel_sizer.Add(self.output_background_choice, 0, wx.EXPAND)

            separator = wx.StaticLine(self.animation_left_panel, -1, size=(256, 5))
            self.animation_left_panel_sizer.Add(separator, 0, wx.EXPAND)

            self.fps_text = wx.StaticText(self.animation_left_panel, label="")
            self.animation_left_panel_sizer.Add(self.fps_text, wx.SizerFlags().Border())

            self.animation_left_panel_sizer.Fit(self.animation_left_panel)

        self.animation_panel_sizer.Fit(self.animation_panel)

    def create_ui(self):
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.main_sizer)
        self.SetAutoLayout(1)

        self.capture_pose_lock = threading.Lock()

        self.create_animation_panel(self)
        self.main_sizer.Add(self.animation_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))

        self.create_capture_panel(self)
        self.main_sizer.Add(self.capture_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))

        self.main_sizer.Fit(self)

    def create_capture_panel(self, parent):
        self.capture_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.capture_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.capture_panel.SetSizer(self.capture_panel_sizer)
        self.capture_panel.SetAutoLayout(1)

        self.webcam_capture_panel = wx.Panel(self.capture_panel, size=(256, 192), style=wx.SIMPLE_BORDER)
        self.webcam_capture_panel.Bind(wx.EVT_PAINT, self.paint_webcam_capture_panel)
        self.webcam_capture_panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
        self.capture_panel_sizer.Add(self.webcam_capture_panel, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 5))

        self.rotation_labels = {}
        self.rotation_value_labels = {}
        rotation_column = self.create_rotation_column(self.capture_panel, HEAD_ROTATIONS)
        self.capture_panel_sizer.Add(rotation_column, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))

    def paint_webcam_capture_panel(self, event: wx.Event):
        wx.BufferedPaintDC(self.webcam_capture_panel, self.webcam_capture_bitmap)

    def create_rotation_column(self, parent, rotation_names):
        column_panel = wx.Panel(parent, style=wx.SIMPLE_BORDER)
        column_panel_sizer = wx.FlexGridSizer(cols=2)
        column_panel_sizer.AddGrowableCol(1)
        column_panel.SetSizer(column_panel_sizer)
        column_panel.SetAutoLayout(1)

        for rotation_name in rotation_names:
            self.rotation_labels[rotation_name] = wx.StaticText(
                column_panel, label=rotation_name, style=wx.ALIGN_RIGHT)
            column_panel_sizer.Add(self.rotation_labels[rotation_name],
                                   wx.SizerFlags(1).Expand().Border(wx.ALL, 3))

            self.rotation_value_labels[rotation_name] = wx.TextCtrl(
                column_panel, style=wx.TE_RIGHT)
            self.rotation_value_labels[rotation_name].SetValue("0.00")
            self.rotation_value_labels[rotation_name].Disable()
            column_panel_sizer.Add(self.rotation_value_labels[rotation_name],
                                   wx.SizerFlags(1).Expand().Border(wx.ALL, 3))

        column_panel.GetSizer().Fit(column_panel)
        return column_panel

    def update_capture_panel(self, event: wx.Event):
        there_is_frame, frame = self.video_capture.read()
        if not there_is_frame:
            dc = wx.MemoryDC()
            dc.SelectObject(self.webcam_capture_bitmap)
            self.draw_nothing_yet_string(dc)
            del dc
            return

        rgb_frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
        resized_frame = cv2.resize(rgb_frame, (256, 192))
        wx_image = wx.ImageFromBuffer(256, 192, resized_frame.tobytes())
        wx_bitmap = wx_image.ConvertToBitmap()

        dc = wx.MemoryDC()
        dc.SelectObject(self.webcam_capture_bitmap)
        dc.Clear()
        dc.DrawBitmap(wx_bitmap, 0, 0, True)
        del dc

        self.webcam_capture_panel.Refresh()

        time_ms = int(time.time() * 1000)
        mediapipe_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.face_landmarker.detect_for_video(mediapipe_image, time_ms)
        self.update_mediapipe_face_pose(detection_result)

    def update_mediapipe_face_pose(self, detection_result):
        if len(detection_result.facial_transformation_matrixes) == 0:
            return

        xform_matrix = detection_result.facial_transformation_matrixes[0]
        blendshape_params = {}
        for item in detection_result.face_blendshapes[0]:
            blendshape_params[item.category_name] = item.score
        M = xform_matrix[0:3, 0:3]
        rot = Rotation.from_matrix(M)
        euler_angles = rot.as_euler('xyz', degrees=True)

        self.rotation_value_labels[HEAD_X].SetValue("%0.2f" % euler_angles[0])
        self.rotation_value_labels[HEAD_X].Refresh()
        self.rotation_value_labels[HEAD_Y].SetValue("%0.2f" % euler_angles[1])
        self.rotation_value_labels[HEAD_Y].Refresh()
        self.rotation_value_labels[HEAD_Z].SetValue("%0.2f" % euler_angles[2])
        self.rotation_value_labels[HEAD_Z].Refresh()

        self.mediapipe_face_pose = MediaPipeFacePose(blendshape_params, xform_matrix)

    @staticmethod
    def convert_to_100(x):
        return int(max(0.0, min(1.0, x)) * 100)

    def paint_source_image_panel(self, event: wx.Event):
        wx.BufferedPaintDC(self.source_image_panel, self.source_image_bitmap)

    def update_source_image_bitmap(self):
        dc = wx.MemoryDC()
        dc.SelectObject(self.source_image_bitmap)
        if self.wx_source_image is None:
            self.draw_nothing_yet_string(dc)
        else:
            dc.Clear()
            dc.DrawBitmap(self.wx_source_image, 0, 0, True)
        del dc

    def draw_nothing_yet_string(self, dc):
        dc.Clear()
        font = wx.Font(wx.FontInfo(14).Family(wx.FONTFAMILY_SWISS))
        dc.SetFont(font)
        w, h = dc.GetTextExtent("Nothing yet!")
        dc.DrawText("Nothing yet!", (MainFrame.IMAGE_SIZE - w) // 2, (MainFrame.IMAGE_SIZE - h) // 2)

    def paint_result_image_panel(self, event: wx.Event):
        wx.BufferedPaintDC(self.result_image_panel, self.result_image_bitmap)

    def update_result_image_bitmap(self, event: Optional[wx.Event] = None):
        if self.mediapipe_face_pose is None or self.poser is None:
            dc = wx.MemoryDC()
            dc.SelectObject(self.result_image_bitmap)
            self.draw_nothing_yet_string(dc)
            del dc
            return

        current_pose = self.pose_converter.convert(self.mediapipe_face_pose)
        if self.last_pose is not None and self.last_pose == current_pose:
            return
        self.last_pose = current_pose

        if self.torch_source_image is None:
            dc = wx.MemoryDC()
            dc.SelectObject(self.result_image_bitmap)
            self.draw_nothing_yet_string(dc)
            del dc
            return

        pose = torch.tensor(current_pose, device=self.device, dtype=self.poser.get_dtype())

        with torch.no_grad():
            output_image = self.poser.pose(self.torch_source_image, pose)[0].float()
            output_image = torch.clip((output_image + 1.0) / 2.0, 0.0, 1.0)
            output_image = convert_linear_to_srgb(output_image)

            background_choice = self.output_background_choice.GetSelection()
            if background_choice == 0:
                pass
            else:
                background = torch.zeros(4, output_image.shape[1], output_image.shape[2], device=self.device)
                background[3, :, :] = 1.0
                if background_choice == 1:
                    background[1, :, :] = 1.0
                    output_image = self.blend_with_background(output_image, background)
                elif background_choice == 2:
                    background[2, :, :] = 1.0
                    output_image = self.blend_with_background(output_image, background)
                elif background_choice == 3:
                    output_image = self.blend_with_background(output_image, background)
                else:
                    background[0:3, :, :] = 1.0
                    output_image = self.blend_with_background(output_image, background)

            c, h, w = output_image.shape
            output_image = 255.0 * torch.transpose(output_image.reshape(c, h * w), 0, 1).reshape(h, w, c)
            output_image = output_image.byte()

        numpy_image = output_image.detach().cpu().numpy()
        wx_image = wx.ImageFromBuffer(numpy_image.shape[0],
                                      numpy_image.shape[1],
                                      numpy_image[:, :, 0:3].tobytes(),
                                      numpy_image[:, :, 3].tobytes())
        wx_bitmap = wx_image.ConvertToBitmap()

        dc = wx.MemoryDC()
        dc.SelectObject(self.result_image_bitmap)
        dc.Clear()
        dc.DrawBitmap(wx_bitmap,
                      (MainFrame.IMAGE_SIZE - numpy_image.shape[0]) // 2,
                      (MainFrame.IMAGE_SIZE - numpy_image.shape[1]) // 2, True)
        del dc

        time_now = time.time_ns()
        if self.last_update_time is not None:
            elapsed_time = time_now - self.last_update_time
            fps = 1.0 / (elapsed_time / 10 ** 9)
            if self.torch_source_image is not None:
                self.fps_statistics.add_fps(fps)
            self.fps_text.SetLabelText("FPS = %0.2f" % self.fps_statistics.get_average_fps())
        self.last_update_time = time_now

        self.Refresh()

    def blend_with_background(self, numpy_image, background):
        alpha = numpy_image[3:4, :, :]
        color = numpy_image[0:3, :, :]
        new_color = color * alpha + (1.0 - alpha) * background[0:3, :, :]
        return torch.cat([new_color, background[3:4, :, :]], dim=0)

    def load_model(self, event: wx.Event):
        dir_name = "data/character_models"
        file_dialog = wx.FileDialog(self, "Choose a model", dir_name, "", "*.yaml", wx.FD_OPEN)
        if file_dialog.ShowModal() == wx.ID_OK:
            character_model_json_file_name = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
            try:
                self.character_model = CharacterModel.load(character_model_json_file_name)
                self.torch_source_image = self.character_model.get_character_image(self.device)
                pil_image = resize_PIL_image(
                    PIL.Image.open(self.character_model.character_image_file_name),
                    (MainFrame.IMAGE_SIZE, MainFrame.IMAGE_SIZE))
                w, h = pil_image.size
                self.wx_source_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                self.update_source_image_bitmap()
                self.poser = self.character_model.get_poser(self.device)
            except Exception:
                message_dialog = wx.MessageDialog(
                    self, "Could not load character model " + character_model_json_file_name, "Poser", wx.OK)
                message_dialog.ShowModal()
                message_dialog.Destroy()
        file_dialog.Destroy()
        self.Refresh()


if __name__ == "__main__":
    device = torch.device("cuda:0")

    pose_converter = MediaPoseFacePoseConverter00()

    face_landmarker_base_options = mediapipe.tasks.BaseOptions(
        model_asset_path='data/thirdparty/mediapipe/face_landmarker_v2_with_blendshapes.task')
    options = mediapipe.tasks.vision.FaceLandmarkerOptions(
        base_options=face_landmarker_base_options,
        running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1)
    face_landmarker = mediapipe.tasks.vision.FaceLandmarker.create_from_options(options)

    video_capture = cv2.VideoCapture(0)

    app = wx.App()
    main_frame = MainFrame(pose_converter, video_capture, face_landmarker, device)
    main_frame.Show(True)
    main_frame.capture_timer.Start(30)
    main_frame.animation_timer.Start(30)
    app.MainLoop()

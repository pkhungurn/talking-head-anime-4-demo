import multiprocessing
import random
from contextlib import contextmanager
from typing import Callable
import PIL.Image

import torch
import wx
import wx.html
import wx.lib.intctrl
from tha4.distiller.ui.distiller_config_state import DistillerConfigState
from tha4.image_util import convert_output_image_from_torch_to_numpy
from tha4.shion.base.image_util import extract_pytorch_image_from_PIL_image


def wx_bind_event(widget, evt):
    def f(handler):
        widget.Bind(evt, handler)
        return handler

    return f


class DistillerUiMainFrame(wx.Frame):
    PARAM_NAME_STATIC_TEXT_MIN_WIDTH = 400
    NUM_TRAINING_EXAMPLES_PER_SAMPLE_OUTPUT_CHOICES = [
        "10_000", "100_000", "1_000_000", "Do not generate sample outputs"]

    def __init__(self):
        super().__init__(None, wx.ID_ANY, "Distiller UI")

        self.init_ui()
        self.init_menus()
        self.init_bitmaps()
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.state = DistillerConfigState()
        self.update_ui()

        self.config_file_to_run = None

    def init_ui(self):
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.SetSizer(main_sizer)
        self.SetAutoLayout(1)

        left_panel = self.init_left_panel(self)
        main_sizer.Add(left_panel, 0, wx.FIXED_MINSIZE)

        middle_panel = self.init_middle_panel(self)
        main_sizer.Add(middle_panel, 0, wx.EXPAND)

        right_panel = self.init_right_panel(self)
        main_sizer.Add(right_panel, 1, wx.EXPAND)

        main_sizer.Fit(self)

    def init_menus(self):
        self.file_menu = wx.Menu()

        self.new_menu_id = wx.Window.NewControlId()
        self.file_menu.Append(
            self.new_menu_id, item="&New\tCTRL+N", helpString="Create a new distiller configuration.")
        self.Bind(wx.EVT_MENU, self.on_new, id=self.new_menu_id)

        self.open_menu_id = wx.Window.NewControlId()
        self.file_menu.Append(
            self.open_menu_id, item="&Open\tCTRL+O", helpString="Open a distiller confuguration.")
        self.Bind(wx.EVT_MENU, self.on_open, id=self.open_menu_id)

        self.save_menu_id = wx.Window.NewControlId()
        self.save_menu_item = wx.MenuItem(
            self.file_menu, id=self.save_menu_id, text="&Save\tCTRL+S",
            helpString="Save the current distiller configuration. Error message will be shown it it is not well formed.")
        self.Bind(wx.EVT_MENU, self.on_save, id=self.save_menu_id)
        self.file_menu.Append(self.save_menu_item)

        self.file_menu.AppendSeparator()

        self.exit_menu_id = wx.ID_EXIT
        self.file_menu.Append(
            self.exit_menu_id, item="E&xit\tCTRL+Q", helpString="Exit the application.")
        self.Bind(wx.EVT_MENU, self.on_close, id=self.exit_menu_id)

        self.menu_bar = wx.MenuBar()
        self.menu_bar.Append(self.file_menu, "&File")

        self.SetMenuBar(self.menu_bar)

    def init_bitmaps(self):
        self.face_image_bitmap = wx.Bitmap(128, 128)
        self.face_image_pytorch = None
        self.face_mask_image_bitmap = wx.Bitmap(128, 128)
        self.face_mask_image_pytorch = None
        self.mask_on_face_image_bitmap = wx.Bitmap(128, 128)
        self.draw_nothing_yet_string_to_bitmap(self.face_image_bitmap, 128, 128)
        self.draw_nothing_yet_string_to_bitmap(self.face_mask_image_bitmap, 128, 128)
        self.draw_nothing_yet_string_to_bitmap(self.mask_on_face_image_bitmap, 128, 128)

    @contextmanager
    def create_panel(self, parent, sizer, *args, **kwargs):
        panel = wx.Panel(parent, *args, **kwargs)
        panel.SetSizer(sizer)
        panel.SetAutoLayout(1)

        try:
            yield panel, sizer
        finally:
            sizer.Fit(panel)

    def init_left_panel(self, parent):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.BORDER_SIMPLE) as (panel, sizer):
            self.face_image_panel = wx.Panel(panel, size=(128, 128), style=wx.SIMPLE_BORDER)
            self.face_image_panel.Bind(wx.EVT_PAINT, self.on_face_image_panel_paint)
            sizer.Add(self.face_image_panel, 0, wx.EXPAND)

            static_text = wx.StaticText(panel, label="Face", style=wx.ALIGN_CENTER)
            sizer.Add(static_text, 0, wx.EXPAND)

            self.face_mask_image_panel = wx.Panel(panel, size=(128, 128), style=wx.SIMPLE_BORDER)
            self.face_mask_image_panel.Bind(wx.EVT_PAINT, self.on_face_mask_image_panel_paint)
            sizer.Add(self.face_mask_image_panel, 0, wx.EXPAND)

            static_text = wx.StaticText(panel, label="Face mask", style=wx.ALIGN_CENTER)
            sizer.Add(static_text, 0, wx.EXPAND)

            self.mask_on_face_image_panel = wx.Panel(panel, size=(128, 128), style=wx.SIMPLE_BORDER)
            self.mask_on_face_image_panel.Bind(wx.EVT_PAINT, self.on_mask_on_face_image_panel_paint)
            sizer.Add(self.mask_on_face_image_panel, 0, wx.EXPAND)

            static_text = wx.StaticText(panel, label="Mask upon face", style=wx.ALIGN_CENTER)
            sizer.Add(static_text, 0, wx.EXPAND)

        return panel

    def on_erase_background(self, event):
        pass

    def on_face_image_panel_paint(self, event):
        wx.BufferedPaintDC(self.face_image_panel, self.face_image_bitmap)

    def on_face_mask_image_panel_paint(self, event):
        wx.BufferedPaintDC(self.face_mask_image_panel, self.face_mask_image_bitmap)

    def on_mask_on_face_image_panel_paint(self, event):
        wx.BufferedPaintDC(self.mask_on_face_image_panel, self.mask_on_face_image_bitmap)

    def init_middle_panel(self, parent):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.BORDER_SIMPLE) as (panel, sizer):
            sizer.Add(self.init_prefix_panel(panel), 0, wx.EXPAND)
            sizer.Add(self.init_character_image_file_name_panel(panel), 0, wx.EXPAND)
            sizer.Add(self.init_face_mask_image_file_name_panel(panel), 0, wx.EXPAND)
            sizer.Add(self.init_num_cpu_workers_panel(panel), 0, wx.EXPAND)
            sizer.Add(self.init_num_gpus_panel(panel), 0, wx.EXPAND)
            sizer.Add(self.init_face_morpher_random_seed_0_panel(panel), 0, wx.EXPAND)
            sizer.Add(self.init_face_morpher_random_seed_1_panel(panel), 0, wx.EXPAND)
            sizer.Add(self.init_face_morpher_batch_size_panel(panel), 0, wx.EXPAND)
            sizer.Add(self.init_body_morpher_random_seed_0_panel(panel), 0, wx.EXPAND)
            sizer.Add(self.init_body_morpher_random_seed_1_panel(panel), 0, wx.EXPAND)
            sizer.Add(self.init_body_morpher_batch_size_panel(panel), 0, wx.EXPAND)
            sizer.Add(self.init_num_training_examples_per_sample_output_panel(panel), 0, wx.EXPAND)

            self.run_button = wx.Button(panel, label="RUN")
            self.run_button.SetMinSize((-1, 64))
            self.run_button.Bind(wx.EVT_BUTTON, self.on_run)
            sizer.Add(self.run_button, 1, wx.EXPAND)

        return panel

    def init_prefix_panel(self, parent):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.BORDER_SIMPLE) as (panel, panel_sizer):
            prefix_param_name_panel = self.create_param_name_panel_with_help_button(
                panel,
                "prefix (i.e. project directory)",
                self.create_help_button_func("distiller-ui-doc/params/prefix.html"))
            panel_sizer.Add(prefix_param_name_panel, 1, wx.EXPAND)

            with self.create_panel(panel, wx.BoxSizer(wx.HORIZONTAL), style=wx.BORDER_NONE) \
                    as (prefix_panel, prefix_sizer):
                self.prefix_text_ctrl = wx.TextCtrl(prefix_panel, value="")
                self.prefix_text_ctrl.SetEditable(False)
                prefix_sizer.Add(self.prefix_text_ctrl, 1, wx.EXPAND)

                self.prefix_change_button = wx.Button(prefix_panel, label="Change...")
                self.prefix_change_button.Bind(wx.EVT_BUTTON, self.on_prefix_change_button)
                prefix_sizer.Add(self.prefix_change_button, 0, wx.EXPAND)
            panel_sizer.Add(prefix_panel, 1, wx.EXPAND)

        return panel

    def on_prefix_change_button(self, event):
        dir_dialog = wx.DirDialog(self, "Choose a directory.", style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
        if dir_dialog.ShowModal() != wx.ID_OK:
            return
        prefix_value = dir_dialog.GetPath()
        try:
            self.state.set_prefix(prefix_value)
            self.update_ui()
        except Exception as e:
            message_dialog = wx.MessageDialog(self, str(e), "Error", wx.OK | wx.ICON_ERROR)
            message_dialog.ShowModal()

    def init_character_image_file_name_panel(self, parent):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.BORDER_SIMPLE) as (panel, panel_sizer):
            prefix_param_name_panel = self.create_param_name_panel_with_help_button(
                panel,
                "character_image_file_name",
                self.create_help_button_func("distiller-ui-doc/params/character_image_file_name.html"))
            panel_sizer.Add(prefix_param_name_panel, 1, wx.EXPAND)

            with self.create_panel(panel, wx.BoxSizer(wx.HORIZONTAL), style=wx.BORDER_NONE) as (sub_panel, sub_sizer):
                self.character_image_file_name_text_ctrl = wx.TextCtrl(sub_panel, value="")
                self.character_image_file_name_text_ctrl.SetEditable(False)
                sub_sizer.Add(self.character_image_file_name_text_ctrl, 1, wx.EXPAND)

                self.character_image_change_button = wx.Button(sub_panel, label="Change...")
                self.character_image_change_button.Bind(wx.EVT_BUTTON, self.on_character_image_change_button)
                sub_sizer.Add(self.character_image_change_button, 0, wx.EXPAND)
            panel_sizer.Add(sub_panel, 1, wx.EXPAND)

        return panel

    def on_character_image_change_button(self, event):
        file_dialog = wx.FileDialog(self, "Choose a PNG file", wildcard="*.png", style=wx.FD_OPEN)
        if file_dialog.ShowModal() != wx.ID_OK:
            return
        file_name = file_dialog.GetPath()
        try:
            self.state.set_character_image_file_name(file_name)
            self.update_face_image_bitmap(file_name)
            self.update_ui()
        except Exception as e:
            message_dialog = wx.MessageDialog(self, str(e), "Error", wx.OK | wx.ICON_ERROR)
            message_dialog.ShowModal()

    def update_face_image_bitmap(self, new_file_name: str):
        pil_image = PIL.Image.open(new_file_name)
        subimage = pil_image.crop((256 - 64, 80, 256 + 64, 208))
        self.face_image_bitmap = wx.Bitmap.FromBufferRGBA(128, 128, subimage.convert("RGBA").tobytes())
        self.face_image_pytorch = extract_pytorch_image_from_PIL_image(subimage).to(torch.float)
        self.update_mask_on_face_image_bitmap()

    def init_face_mask_image_file_name_panel(self, parent):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.BORDER_SIMPLE) as (panel, panel_sizer):
            prefix_param_name_panel = self.create_param_name_panel_with_help_button(
                panel,
                "face_mask_image_file_name",
                self.create_help_button_func("distiller-ui-doc/params/face_mask_image_file_name.html"))
            panel_sizer.Add(prefix_param_name_panel, 1, wx.EXPAND)

            with self.create_panel(panel, wx.BoxSizer(wx.HORIZONTAL), style=wx.BORDER_NONE) as (sub_panel, sub_sizer):
                self.face_mask_image_file_name_text_ctrl = wx.TextCtrl(sub_panel, value="")
                self.face_mask_image_file_name_text_ctrl.SetEditable(False)
                sub_sizer.Add(self.face_mask_image_file_name_text_ctrl, 1, wx.EXPAND)

                self.face_mask_image_file_name_change_button = wx.Button(sub_panel, label="Change...")
                self.face_mask_image_file_name_change_button.Bind(wx.EVT_BUTTON, self.on_face_mask_image_change_button)
                sub_sizer.Add(self.face_mask_image_file_name_change_button, 0, wx.EXPAND)
            panel_sizer.Add(sub_panel, 1, wx.EXPAND)

        return panel

    def on_face_mask_image_change_button(self, event):
        file_dialog = wx.FileDialog(self, "Choose a PNG file", wildcard="*.png", style=wx.FD_OPEN)
        if file_dialog.ShowModal() != wx.ID_OK:
            return
        file_name = file_dialog.GetPath()
        try:
            self.state.set_face_mask_image_file_name(file_name)
            self.update_face_mask_image_bitmap(file_name)
            self.update_ui()
        except Exception as e:
            message_dialog = wx.MessageDialog(self, str(e), "Error", wx.OK | wx.ICON_ERROR)
            message_dialog.ShowModal()

    def update_face_mask_image_bitmap(self, new_file_name):
        pil_image = PIL.Image.open(new_file_name)
        subimage = pil_image.crop((256 - 64, 80, 256 + 64, 208))
        self.face_mask_image_bitmap = wx.Bitmap.FromBufferRGBA(128, 128, subimage.convert("RGBA").tobytes())
        self.face_mask_image_pytorch = extract_pytorch_image_from_PIL_image(subimage).to(torch.float)
        self.face_mask_image_pytorch = self.face_mask_image_pytorch[0:1, :, :]
        self.update_mask_on_face_image_bitmap()

    def update_mask_on_face_image_bitmap(self):
        if self.face_image_pytorch is None:
            return
        if self.face_mask_image_pytorch is None:
            return

        mask_on_face_image = (0.5 * self.face_image_pytorch) + (0.5 * self.face_mask_image_pytorch)
        numpy_image = convert_output_image_from_torch_to_numpy(mask_on_face_image)
        wx_image = wx.ImageFromBuffer(
            numpy_image.shape[0],
            numpy_image.shape[1],
            numpy_image[:, :, 0:3].tobytes(),
            numpy_image[:, :, 3].tobytes())
        self.mask_on_face_image_bitmap = wx_image.ConvertToBitmap()

    def init_num_cpu_workers_panel(self, parent):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.BORDER_SIMPLE) as (panel, panel_sizer):
            prefix_param_name_panel = self.create_param_name_panel_with_help_button(
                panel,
                "num_cpu_workers",
                self.create_help_button_func("distiller-ui-doc/params/num_cpu_workers.html"))
            panel_sizer.Add(prefix_param_name_panel, 1, wx.EXPAND)

            num_cpus = multiprocessing.cpu_count()
            self.num_cpu_workers_spin_ctrl = wx.SpinCtrl(panel, initial=1, min=1, max=num_cpus)

            @wx_bind_event(self.num_cpu_workers_spin_ctrl, wx.EVT_SPINCTRL)
            def on_num_cpu_workers_spin_ctrl(event):
                self.state.set_num_cpu_workers(self.num_cpu_workers_spin_ctrl.GetValue())
                self.Refresh()

            panel_sizer.Add(self.num_cpu_workers_spin_ctrl, 1, wx.EXPAND)

        return panel

    def init_num_gpus_panel(self, parent):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.BORDER_SIMPLE) as (panel, panel_sizer):
            prefix_param_name_panel = self.create_param_name_panel_with_help_button(
                panel,
                "num_gpus",
                self.create_help_button_func("distiller-ui-doc/params/num_gpus.html"))
            panel_sizer.Add(prefix_param_name_panel, 1, wx.EXPAND)

            num_gpus = torch.cuda.device_count()
            self.num_gpus_spin_ctrl = wx.SpinCtrl(panel, initial=1, min=1, max=max(1, num_gpus))

            @wx_bind_event(self.num_gpus_spin_ctrl, wx.EVT_SPINCTRL)
            def on_num_cpu_workers_spin_ctrl(event):
                self.state.set_num_gpus(self.num_gpus_spin_ctrl.GetValue())
                self.Refresh()

            panel_sizer.Add(self.num_gpus_spin_ctrl, 1, wx.EXPAND)

        return panel

    def init_face_morpher_random_seed_0_panel(self, parent):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.BORDER_SIMPLE) as (panel, panel_sizer):
            prefix_param_name_panel = self.create_param_name_panel_with_help_button(
                panel,
                "face_morpher_random_seed_0",
                self.create_help_button_func("distiller-ui-doc/params/face_morpher_random_seed_0.html"))
            panel_sizer.Add(prefix_param_name_panel, 1, wx.EXPAND)

            with self.create_panel(panel, wx.BoxSizer(wx.HORIZONTAL), style=wx.BORDER_NONE) as (sub_panel, sub_sizer):
                initial_value = random.randint(0, 2 ** 64 - 1)
                self.face_morpher_random_seed_0_int_ctrl = wx.lib.intctrl.IntCtrl(
                    sub_panel, value=initial_value, min=0, max=0x_ffff_ffff_ffff_ffff)

                @wx_bind_event(self.face_morpher_random_seed_0_int_ctrl, wx.EVT_TEXT)
                def on_face_morpher_random_seed_0_int_ctrl_text(event):
                    self.state.set_face_morpher_random_seed_0(self.face_morpher_random_seed_0_int_ctrl.GetValue())

                sub_sizer.Add(self.face_morpher_random_seed_0_int_ctrl, 1, wx.EXPAND)

                self.face_morpher_random_seed_0_randomize_button = wx.Button(sub_panel, label="Randomize")

                @wx_bind_event(self.face_morpher_random_seed_0_randomize_button, wx.EVT_BUTTON)
                def on_face_morpher_random_seed_0_randomize_button(event):
                    new_value = random.randint(0, 0x_ffff_ffff_ffff_ffff)
                    self.face_morpher_random_seed_0_int_ctrl.SetValue(new_value)
                    self.state.set_face_morpher_random_seed_0(new_value)

                sub_sizer.Add(self.face_morpher_random_seed_0_randomize_button, 0, wx.EXPAND)
            panel_sizer.Add(sub_panel, 1, wx.EXPAND)

        return panel

    def init_face_morpher_random_seed_1_panel(self, parent):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.BORDER_SIMPLE) as (panel, panel_sizer):
            prefix_param_name_panel = self.create_param_name_panel_with_help_button(
                panel,
                "face_morpher_random_seed_1",
                self.create_help_button_func("distiller-ui-doc/params/face_morpher_random_seed_1.html"))
            panel_sizer.Add(prefix_param_name_panel, 1, wx.EXPAND)

            with self.create_panel(panel, wx.BoxSizer(wx.HORIZONTAL), style=wx.BORDER_NONE) as (sub_panel, sub_sizer):
                initial_value = random.randint(0, 2 ** 64 - 1)
                self.face_morpher_random_seed_1_int_ctrl = wx.lib.intctrl.IntCtrl(
                    sub_panel, value=initial_value, min=0, max=0x_ffff_ffff_ffff_ffff)

                @wx_bind_event(self.face_morpher_random_seed_1_int_ctrl, wx.EVT_TEXT)
                def on_face_morpher_random_seed_1_int_ctrl_text(event):
                    self.state.set_face_morpher_random_seed_1(self.face_morpher_random_seed_1_int_ctrl.GetValue())

                sub_sizer.Add(self.face_morpher_random_seed_1_int_ctrl, 1, wx.EXPAND)

                self.face_morpher_random_seed_1_randomize_button = wx.Button(sub_panel, label="Randomize")

                @wx_bind_event(self.face_morpher_random_seed_1_randomize_button, wx.EVT_BUTTON)
                def on_face_morpher_random_seed_1_randomize_button(event):
                    new_value = random.randint(0, 0x_ffff_ffff_ffff_ffff)
                    self.face_morpher_random_seed_1_int_ctrl.SetValue(new_value)
                    self.state.set_face_morpher_random_seed_1(new_value)

                sub_sizer.Add(self.face_morpher_random_seed_1_randomize_button, 0, wx.EXPAND)
            panel_sizer.Add(sub_panel, 1, wx.EXPAND)

        return panel

    def init_face_morpher_batch_size_panel(self, parent):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.BORDER_SIMPLE) as (panel, panel_sizer):
            prefix_param_name_panel = self.create_param_name_panel_with_help_button(
                panel,
                "face_morpher_batch_size",
                self.create_help_button_func("distiller-ui-doc/params/face_morpher_batch_size.html"))
            panel_sizer.Add(prefix_param_name_panel, 1, wx.EXPAND)

            self.face_morpher_batch_size_spin_ctrl = wx.SpinCtrl(panel, initial=8, min=1, max=8)

            @wx_bind_event(self.face_morpher_batch_size_spin_ctrl, wx.EVT_SPINCTRL)
            def on_face_morpher_batch_size_spin_ctrl(event):
                self.state.set_face_morpher_batch_size(self.face_morpher_batch_size_spin_ctrl.GetValue())

            panel_sizer.Add(self.face_morpher_batch_size_spin_ctrl, 1, wx.EXPAND)

        return panel

    def init_body_morpher_random_seed_0_panel(self, parent):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.BORDER_SIMPLE) as (panel, panel_sizer):
            prefix_param_name_panel = self.create_param_name_panel_with_help_button(
                panel,
                "body_morpher_random_seed_0",
                self.create_help_button_func("distiller-ui-doc/params/body_morpher_random_seed_0.html"))
            panel_sizer.Add(prefix_param_name_panel, 1, wx.EXPAND)

            with self.create_panel(panel, wx.BoxSizer(wx.HORIZONTAL), style=wx.BORDER_NONE) as (sub_panel, sub_sizer):
                initial_value = random.randint(0, 2 ** 64 - 1)
                self.body_morpher_random_seed_0_int_ctrl = wx.lib.intctrl.IntCtrl(
                    sub_panel, value=initial_value, min=0, max=0x_ffff_ffff_ffff_ffff)

                @wx_bind_event(self.body_morpher_random_seed_0_int_ctrl, wx.EVT_TEXT)
                def on_body_morpher_random_seed_0_int_ctrl_text(event):
                    self.state.set_body_morpher_random_seed_0(self.body_morpher_random_seed_0_int_ctrl.GetValue())

                sub_sizer.Add(self.body_morpher_random_seed_0_int_ctrl, 1, wx.EXPAND)

                self.body_morpher_random_seed_0_randomize_button = wx.Button(sub_panel, label="Randomize")

                @wx_bind_event(self.body_morpher_random_seed_0_randomize_button, wx.EVT_BUTTON)
                def on_body_morpher_random_seed_0_randomize_button(event):
                    new_value = random.randint(0, 0x_ffff_ffff_ffff_ffff)
                    self.body_morpher_random_seed_0_int_ctrl.SetValue(new_value)
                    self.state.set_body_morpher_random_seed_0(new_value)

                sub_sizer.Add(self.body_morpher_random_seed_0_randomize_button, 0, wx.EXPAND)
            panel_sizer.Add(sub_panel, 1, wx.EXPAND)

        return panel

    def init_body_morpher_random_seed_1_panel(self, parent):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.BORDER_SIMPLE) as (panel, panel_sizer):
            prefix_param_name_panel = self.create_param_name_panel_with_help_button(
                panel,
                "body_morpher_random_seed_1",
                self.create_help_button_func("distiller-ui-doc/params/body_morpher_random_seed_1.html"))
            panel_sizer.Add(prefix_param_name_panel, 1, wx.EXPAND)

            with self.create_panel(panel, wx.BoxSizer(wx.HORIZONTAL), style=wx.BORDER_NONE) as (sub_panel, sub_sizer):
                initial_value = random.randint(0, 2 ** 64 - 1)
                self.body_morpher_random_seed_1_int_ctrl = wx.lib.intctrl.IntCtrl(
                    sub_panel, value=initial_value, min=0, max=0x_ffff_ffff_ffff_ffff)

                @wx_bind_event(self.body_morpher_random_seed_1_int_ctrl, wx.EVT_TEXT)
                def on_body_morpher_random_seed_1_int_ctrl_text(event):
                    self.state.set_body_morpher_random_seed_1(self.body_morpher_random_seed_1_int_ctrl.GetValue())

                sub_sizer.Add(self.body_morpher_random_seed_1_int_ctrl, 1, wx.EXPAND)

                self.body_morpher_random_seed_1_randomize_button = wx.Button(sub_panel, label="Randomize")

                @wx_bind_event(self.body_morpher_random_seed_1_randomize_button, wx.EVT_BUTTON)
                def on_body_morpher_random_seed_1_randomize_button(event):
                    new_value = random.randint(0, 0x_ffff_ffff_ffff_ffff)
                    self.body_morpher_random_seed_1_int_ctrl.SetValue(new_value)
                    self.state.set_body_morpher_random_seed_1(new_value)

                sub_sizer.Add(self.body_morpher_random_seed_1_randomize_button, 0, wx.EXPAND)
            panel_sizer.Add(sub_panel, 1, wx.EXPAND)

        return panel

    def init_body_morpher_batch_size_panel(self, parent):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.BORDER_SIMPLE) as (panel, panel_sizer):
            prefix_param_name_panel = self.create_param_name_panel_with_help_button(
                panel,
                "body_morpher_batch_size",
                self.create_help_button_func("distiller-ui-doc/params/body_morpher_batch_size.html"))
            panel_sizer.Add(prefix_param_name_panel, 1, wx.EXPAND)

            self.body_morpher_batch_size_spin_ctrl = wx.SpinCtrl(panel, initial=8, min=1, max=8)

            @wx_bind_event(self.body_morpher_batch_size_spin_ctrl, wx.EVT_SPINCTRL)
            def on_body_morpher_batch_size_spin_ctrl(event):
                self.state.set_body_morpher_batch_size(self.body_morpher_batch_size_spin_ctrl.GetValue())

            panel_sizer.Add(self.body_morpher_batch_size_spin_ctrl, 1, wx.EXPAND)

        return panel

    def init_num_training_examples_per_sample_output_panel(self, parent):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.BORDER_SIMPLE) as (panel, panel_sizer):
            prefix_param_name_panel = self.create_param_name_panel_with_help_button(
                panel,
                "num_training_examples_per_sample_output",
                self.create_help_button_func("distiller-ui-doc/params/num_training_examples_per_sample_output.html"))
            panel_sizer.Add(prefix_param_name_panel, 1, wx.EXPAND)

            self.num_training_examples_per_sample_output_combobox = \
                wx.ComboBox(panel,
                            value="10_000",
                            choices=DistillerUiMainFrame.NUM_TRAINING_EXAMPLES_PER_SAMPLE_OUTPUT_CHOICES)

            @wx_bind_event(self.num_training_examples_per_sample_output_combobox, wx.EVT_COMBOBOX)
            def on_num_training_examples_per_sample_output_combobox(event):
                index = self.num_training_examples_per_sample_output_combobox.GetSelection()
                if index == 3:
                    self.state.set_face_morpher_num_training_examples_per_sample_output(None)
                    self.state.set_body_morpher_num_training_examples_per_sample_output(None)
                else:
                    selected = DistillerUiMainFrame.NUM_TRAINING_EXAMPLES_PER_SAMPLE_OUTPUT_CHOICES[index]
                    new_value = int(selected)
                    self.state.set_face_morpher_num_training_examples_per_sample_output(new_value)
                    self.state.set_body_morpher_num_training_examples_per_sample_output(new_value)

            panel_sizer.Add(self.num_training_examples_per_sample_output_combobox, 1, wx.EXPAND)

        return panel

    def on_close(self, event):
        if self.state.dirty:
            confirmation_dialog = wx.MessageDialog(
                parent=self,
                message=f"You have not saved your work. Do you want to exit anyway?",
                caption="Confirmation",
                style=wx.YES_NO | wx.ICON_QUESTION)
            result = confirmation_dialog.ShowModal()
            if result == wx.ID_NO:
                return

        self.Destroy()

    def create_help_button_func(self, html_file_name: str):
        def init_help_button_func(parent):
            button = wx.Button(parent, label="Help")

            @wx_bind_event(button, wx.EVT_BUTTON)
            def on_prefix_button(event):
                self.html_window.LoadPage(html_file_name)
                self.Refresh()

            return button

        return init_help_button_func

    def create_param_name_panel_with_help_button(
            self, parent, param_name: str, help_button_func: Callable[[wx.Window], wx.Button]):
        with self.create_panel(parent, wx.BoxSizer(wx.HORIZONTAL), style=wx.NO_BORDER) \
                as (panel, sizer):
            title_text_panel = self.create_vertically_centered_text_panel(
                panel, param_name, DistillerUiMainFrame.PARAM_NAME_STATIC_TEXT_MIN_WIDTH)
            sizer.Add(title_text_panel, 1, wx.EXPAND)

            help_button = help_button_func(panel)
            sizer.Add(help_button, 0, wx.EXPAND)
        return panel

    def create_vertically_centered_text_panel(self, parent, text: str, min_width: int):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.NO_BORDER) as (panel, sizer):
            sizer.AddStretchSpacer(1)
            text = wx.StaticText(
                panel,
                label=text,
                style=wx.ALIGN_CENTER)
            text.SetMinSize((min_width, -1))
            sizer.Add(text, 0, wx.EXPAND)
            sizer.AddStretchSpacer(1)
        return panel

    def init_right_panel(self, parent):
        with self.create_panel(parent, wx.BoxSizer(wx.VERTICAL), style=wx.BORDER_SIMPLE) as (panel, sizer):
            self.html_window = wx.html.HtmlWindow(panel)
            self.html_window.SetMinSize((600, 600))
            self.html_window.SetFonts("Times New Roman", "Courier New", sizes=[10, 12, 14, 16, 18, 20, 24])
            self.html_window.LoadPage("distiller-ui-doc/index.html")
            sizer.Add(self.html_window, 1, wx.EXPAND)

            go_to_main_documentation_button = wx.Button(panel, label="Go to Main Documentation")
            sizer.Add(go_to_main_documentation_button, 0, wx.EXPAND)

            @wx_bind_event(go_to_main_documentation_button, wx.EVT_BUTTON)
            def on_go_to_main_documentation_button(event):
                self.html_window.LoadPage("distiller-ui-doc/index.html")
                self.Refresh()

        return panel

    def populate_distiller_config(self):
        self.state.config.prefix = self.prefix_text_ctrl.GetValue()
        self.state.config.character_image_file_name = self.character_image_file_name_text_ctrl.GetValue()
        self.state.config.face_mask_image_file_name = self.face_mask_image_file_name_text_ctrl.GetValue()

        self.state.config.num_cpu_workers = self.num_cpu_workers_spin_ctrl.GetValue()
        self.state.config.num_gpus = self.num_gpus_spin_ctrl.GetValue()

        self.state.config.face_morpher_random_seed_0 = self.face_morpher_random_seed_0_int_ctrl.GetValue()
        self.state.config.face_morpher_random_seed_1 = self.face_morpher_random_seed_1_int_ctrl.GetValue()
        self.state.config.face_morpher_batch_size = self.face_morpher_batch_size_spin_ctrl.GetValue()

        self.state.config.body_morpher_random_seed_0 = self.body_morpher_random_seed_0_int_ctrl.GetValue()
        self.state.config.body_morpher_random_seed_1 = self.body_morpher_random_seed_1_int_ctrl.GetValue()
        self.state.config.body_morpher_batch_size = self.body_morpher_batch_size_spin_ctrl.GetValue()

        if self.num_training_examples_per_sample_output_combobox.GetValue() == \
                DistillerUiMainFrame.NUM_TRAINING_EXAMPLES_PER_SAMPLE_OUTPUT_CHOICES[-1]:
            self.state.config.face_morpher_num_training_examples_per_sample_output = None
            self.state.config.body_morpher_num_training_examples_per_sample_output = None
        else:
            value = int(self.num_training_examples_per_sample_output_combobox.GetValue())
            self.state.config.face_morpher_num_training_examples_per_sample_output = value
            self.state.config.body_morpher_num_training_examples_per_sample_output = value

    def update_ui(self):
        self.prefix_text_ctrl.SetValue(self.state.config.prefix)
        self.character_image_file_name_text_ctrl.SetValue(self.state.config.character_image_file_name)
        self.face_mask_image_file_name_text_ctrl.SetValue(self.state.config.face_mask_image_file_name)

        if not self.state.can_show_character_image():
            self.draw_nothing_yet_string_to_bitmap(self.face_image_bitmap, 128, 128)
        if not self.state.can_show_face_mask_image():
            self.draw_nothing_yet_string_to_bitmap(self.face_mask_image_bitmap, 128, 128)
        if not self.state.can_show_mask_on_face_image():
            self.draw_nothing_yet_string_to_bitmap(self.mask_on_face_image_bitmap, 128, 128)

        self.num_cpu_workers_spin_ctrl.SetValue(self.state.config.num_cpu_workers)
        self.num_gpus_spin_ctrl.SetValue(self.state.config.num_gpus)

        self.face_morpher_random_seed_0_int_ctrl.SetValue(self.state.config.face_morpher_random_seed_0)
        self.face_morpher_random_seed_1_int_ctrl.SetValue(self.state.config.face_morpher_random_seed_1)
        self.face_morpher_batch_size_spin_ctrl.SetValue(self.state.config.face_morpher_batch_size)

        self.body_morpher_random_seed_0_int_ctrl.SetValue(self.state.config.body_morpher_random_seed_0)
        self.body_morpher_random_seed_1_int_ctrl.SetValue(self.state.config.body_morpher_random_seed_1)
        self.body_morpher_batch_size_spin_ctrl.SetValue(self.state.config.body_morpher_batch_size)

        if self.state.config.body_morpher_num_training_examples_per_sample_output is None:
            self.num_training_examples_per_sample_output_combobox.SetSelection(3)
        else:
            choices = [int(x) for x in DistillerUiMainFrame.NUM_TRAINING_EXAMPLES_PER_SAMPLE_OUTPUT_CHOICES[:-1]]
            self.num_training_examples_per_sample_output_combobox.SetSelection(
                choices.index(self.state.config.body_morpher_num_training_examples_per_sample_output))

        self.save_menu_item.Enable(self.state.can_save())

        self.Refresh()

    def draw_nothing_yet_string_to_bitmap(self, bitmap, width: int, height: int):
        dc = wx.MemoryDC()
        dc.SelectObject(bitmap)

        dc.Clear()
        font = wx.Font(wx.FontInfo(14).Family(wx.FONTFAMILY_SWISS))
        dc.SetFont(font)
        w, h = dc.GetTextExtent("Nothing yet!")
        dc.DrawText("Nothing yet!", (width - w) // 2, (height - h) // 2)

        del dc

    def try_saving(self):
        if not self.state.can_save():
            message_dialog = wx.MessageDialog(
                self,
                "Cannot save yet! Please make sure you set the prefix, character_image_file_name, "
                "and face_mask_image_file_name first.",
                "Error",
                wx.OK | wx.ICON_ERROR)
            message_dialog.ShowModal()
            return False
        else:
            if self.state.need_to_check_overwrite():
                confirmation_dialog = wx.MessageDialog(
                    parent=self,
                    message=f"Overwriting {self.state.config.config_yaml_file_name()}?",
                    caption="Confirmation",
                    style=wx.YES_NO | wx.CANCEL | wx.ICON_QUESTION)
                result = confirmation_dialog.ShowModal()
                if result == wx.ID_YES:
                    self.state.save()
                    return True
                elif result == wx.ID_NO:
                    return False
                else:
                    return False
            else:
                self.state.save()
                return True

    def on_save(self, event):
        return self.try_saving()

    def on_new(self, event):
        if self.state.dirty:
            confirmation_dialog = wx.MessageDialog(
                parent=self,
                message=f"You have not saved the current config. Do you want to proceed?",
                caption="Confirmation",
                style=wx.YES_NO | wx.ICON_QUESTION)
            result = confirmation_dialog.ShowModal()
            if result == wx.ID_NO:
                return
        self.state = DistillerConfigState()
        self.update_ui()

    def on_open(self, event):
        if self.state.dirty:
            confirmation_dialog = wx.MessageDialog(
                parent=self,
                message=f"You have not saved the current config. Do you want to proceed?",
                caption="Confirmation",
                style=wx.YES_NO | wx.ICON_QUESTION)
            result = confirmation_dialog.ShowModal()
            if result == wx.ID_NO:
                return

        file_dialog = wx.FileDialog(self, "Choose a YAML file", wildcard="*.yaml", style=wx.FD_OPEN)
        if file_dialog.ShowModal() != wx.ID_OK:
            return
        file_name = file_dialog.GetPath()
        try:
            self.state.load(file_name)
            self.face_image_pytorch = None
            self.face_mask_image_pytorch = None
            self.update_face_image_bitmap(self.state.config.character_image_file_name)
            self.update_face_mask_image_bitmap(self.state.config.face_mask_image_file_name)
            self.update_ui()
        except Exception as e:
            message_dialog = wx.MessageDialog(self, str(e), "Error", wx.OK | wx.ICON_ERROR)
            message_dialog.ShowModal()

    def on_run(self, event):
        try:
            self.state.config.check()
        except Exception as e:
            message_dialog = wx.MessageDialog(self, str(e), "Error", wx.OK | wx.ICON_ERROR)
            message_dialog.ShowModal()
            return

        if self.state.dirty:
            message_dialog = wx.MessageDialog(
                self,
                "Please save the configuration first.",
                "Error",
                wx.OK | wx.ICON_ERROR)
            message_dialog.ShowModal()
            return

        self.config_file_to_run = self.state.config.config_yaml_file_name()
        self.Destroy()

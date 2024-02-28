import wx

from tha4.app.distill import run_config
from tha4.distiller.ui.distiller_ui_main_frame import DistillerUiMainFrame

if __name__ == "__main__":
    app = wx.App()
    main_frame = DistillerUiMainFrame()
    main_frame.Show(True)
    app.MainLoop()

    if main_frame.config_file_to_run is not None:
        run_config(main_frame.config_file_to_run)

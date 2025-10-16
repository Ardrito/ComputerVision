import tkinter as tk
import numpy as np
from Capture_Calibrate import getCalibration

class GUI:
    def __init__(self):
        self.window = tk.Tk()

        self.window.geometry("600x800")
        self.window.title("Camera Calibration Matrix Tool")

        self.header_font = ("Times New Roman", 18)
        self.title_font = ("Times New Roman", 16)
        self.body_font = ("Times New Roman", 12)

        self.header = tk.Label(self.window, text="Camera Calibration Matrix Tool", font=self.header_font)
        self.header.pack(pady=10)

        self.description_text = tk.Label(self.window, text="Tool for finding calibration matrix of a camera. \nThis will be saved as a .npz file. \n\nRun 'v4l2-ctl --list-devices' to find device numbers", font=self.body_font)
        self.description_text.pack(pady=10)

        self.input_grid = tk.Frame(self.window)

        self.camera_number_one = tk.IntVar(self.window,value=0)
        self.camera_number_two = tk.IntVar(self.window,value=2)
        self.chess_board_size_x = tk.IntVar(self.window, value=10)
        self.chess_board_size_y = tk.IntVar(self.window, value=7)
        self.square_size = tk.DoubleVar(self.window, value=0.018)
        self.num_images = tk.IntVar(self.window, value=10)
        self.snap_time = tk.DoubleVar(self.window, value=2)
        self.cam_offset_x = tk.DoubleVar(self.window,value=0.133)
        self.cam_offset_y = tk.DoubleVar(self.window,value=0)
        self.cam_offset_z = tk.DoubleVar(self.window,value=0)
        self.frame_width = tk.IntVar(self.window,value=1280)
        self.frame_height = tk.IntVar(self.window,value=720)

        self.camera_number_label_one = tk.Label(self.input_grid, text="First Camera number:")
        self.camera_number_label_two = tk.Label(self.input_grid, text="Second Camera number:\n(-1 if you only want one camera) ")
        self.chessboard_size_label = tk.Label(self.input_grid, text="Chessboard size:\n(width x Height)")
        self.square_size_label = tk.Label(self.input_grid, text="Square size in meters:")
        self.num_images_label = tk.Label(self.input_grid, text="Number of images to use:")
        self.snap_time_label = tk.Label(self.input_grid, text="Time between image captures:")
        self.cam_offset_label = tk.Label(self.input_grid, text="Distance between camera centers\n(Meters)")
        self.cam_offset_x_label = tk.Label(self.input_grid, text="x: ")
        self.cam_offset_y_label = tk.Label(self.input_grid, text="y: ")
        self.cam_offset_z_label = tk.Label(self.input_grid, text="z: ")
        self.cam_resolution_label = tk.Label(self.input_grid, text="Camera resolution:")
        self.frame_width_label = tk.Label(self.input_grid, text="width: ")
        self.frame_height_label = tk.Label(self.input_grid, text="height: ")

        
        self.camera_number_label_one.grid(row=0,column=0, padx=2,pady=10,sticky='w')
        self.camera_number_one_entry = tk.Entry(self.input_grid, textvariable=self.camera_number_one, width=2)
        self.camera_number_one_entry.grid(row=0,column=1, padx=5,pady=10,sticky='w')

        self.camera_number_label_two.grid(row=0,column=2, padx=2,pady=10)
        self.camera_number_two_entry = tk.Entry(self.input_grid, textvariable=self.camera_number_two, width=2)
        self.camera_number_two_entry.grid(row=0,column=3, padx=5,pady=10)

        self.chessboard_size_label.grid(row=1,column=0, padx=2,pady=10,sticky='w')
        self.chess_board_size_x_entry = tk.Entry(self.input_grid, textvariable=self.chess_board_size_x, width=2)
        self.chess_board_size_x_entry.grid(row=1,column=1,pady=10,sticky='w')
        self.chess_board_size_y_entry = tk.Entry(self.input_grid, textvariable=self.chess_board_size_y, width=2)
        self.chess_board_size_y_entry.grid(row=1,column=2,pady=10,sticky='w')

        self.square_size_label.grid(row=2,column=0,pady=10,sticky='w')
        self.square_size_entry = tk.Entry(self.input_grid, textvariable=self.square_size, width=7)
        self.square_size_entry.grid(row=2,column=1,pady=10,sticky='w')

        self.num_images_label.grid(row=3,column=0,pady=10,sticky='w')
        self.num_images_entry = tk.Entry(self.input_grid, textvariable=self.num_images, width=2)
        self.num_images_entry.grid(row=3,column=1,pady=10,sticky='w')

        self.cam_offset_label.grid(row=4,column=0,pady=10,sticky='w')
        self.cam_offset_x_label.grid(row=5,column=0,pady=10,sticky='e')
        self.cam_offset_x_entry = tk.Entry(self.input_grid, textvariable=self.cam_offset_x, width=5)
        self.cam_offset_x_entry.grid(row=5,column=1,pady=10,sticky='w')
        self.cam_offset_y_label.grid(row=6,column=0,pady=10,sticky='e')
        self.cam_offset_y_entry = tk.Entry(self.input_grid, textvariable=self.cam_offset_y, width=5)
        self.cam_offset_y_entry.grid(row=6,column=1,pady=10,sticky='w')
        self.cam_offset_z_label.grid(row=7,column=0,pady=10,sticky='e')
        self.cam_offset_z_entry = tk.Entry(self.input_grid, textvariable=self.cam_offset_z, width=5)
        self.cam_offset_z_entry.grid(row=7,column=1,pady=10,sticky='w')

        self.snap_time_label.grid(row=8,column=0,pady=15,sticky='w')
        self.snap_time_entry = tk.Entry(self.input_grid, textvariable=self.snap_time, width=3)
        self.snap_time_entry.grid(row=8,column=1,pady=15,sticky='w')

        self.cam_resolution_label.grid(row=9,column=0,pady=10,sticky='w')
        self.frame_width_label.grid(row=10,column=0,pady=10,sticky='e')
        self.frame_width_entry = tk.Entry(self.input_grid, textvariable=self.frame_width, width=5)
        self.frame_width_entry.grid(row=10,column=1,pady=10,sticky='w')
        self.frame_height_label.grid(row=11,column=0,pady=10,sticky='e')
        self.frame_height_entry = tk.Entry(self.input_grid, textvariable=self.frame_height, width=5)
        self.frame_height_entry.grid(row=11,column=1,pady=10,sticky='w')
                
        self.input_grid.pack()


        self.start_button = tk.Button(self.window, text="Start", font=self.body_font, command=self.start)
        self.start_button.pack(pady=10)

        self.window.mainloop()

    def start(self):
        print ("Start:", self.camera_number_one.get())
        camera_matrix_one, dist_one, newMtx_one, roi_one, mapx_one, mapy_one, error_one = getCalibration(
            camera_number=self.camera_number_one.get(),chessboardSize=(self.chess_board_size_x.get(),self.chess_board_size_y.get()),
            square_size=self.square_size.get(),snap_time=self.snap_time.get(), num_images=self.num_images.get(),resolution=(self.frame_width.get(),self.frame_height.get()))
        print ("Camera One Error", error_one)
        print ("Camera One Matrix:", camera_matrix_one)
        
        if self.camera_number_two.get() > -1:
            camera_matrix_two, dist_two, newMtx_two, roi_two, mapx_two, mapy_two, error_two = getCalibration(
            camera_number=self.camera_number_two.get(),chessboardSize=(self.chess_board_size_x.get(),self.chess_board_size_y.get()),
            square_size=self.square_size.get(),snap_time=self.snap_time.get(), num_images=self.num_images.get(),resolution=(self.frame_width.get(),self.frame_height.get()))
        else:
            camera_matrix_two, dist_two, newMtx_two, roi_two, mapx_two, mapy_two, error_two = None , None, None, None, None, None, None


        print ("Camera Two Error", error_two)
        print ("Camera Two Matrix:", camera_matrix_two)

        R = np.eye(3)
        T = np.array([self.cam_offset_x.get(),self.cam_offset_y.get(),self.cam_offset_z.get()])

        print (T)

        np.savez(
            "camera_calibration_data.npz",
            camera_matrix_one = camera_matrix_one,
            dist_one = dist_one,
            newMtx_one = newMtx_one,
            roi_one = roi_one,
            mapx_one = mapx_one,
            mapy_one = mapy_one,
            error_one = error_one,
            camera_matrix_two = camera_matrix_two,
            dist_two = dist_two,
            newMtx_two = newMtx_two,
            roi_two = roi_two,
            mapx_two = mapx_two,
            mapy_two = mapy_two,
            error_two = error_two,
            R=R,
            T=T,
            resolution = (self.frame_width.get(), self.frame_height.get())
        )

        self.window.destroy()

GUI()

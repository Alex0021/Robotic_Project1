####################################################
# THIS FILE CONTAINS THE GUI OF THE SIMULATOR
# FOR MY SEMESTER PROJECT
####################################################

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font
import sys

w,h = (1600,800)


class myApp(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.mainframe = root
        self.mainframe.grid_columnconfigure(0, weight=1)
        self.mainframe.grid_columnconfigure(1, weight=3)
        self.mainframe.grid_rowconfigure(0,weight=1)
        self.init_main_panels()
        self.init_sidebar_components()

    def init_main_panels(self):
        self.panel_view = tk.Frame(self.mainframe)
        self.panel_view.grid_columnconfigure(0,weight=1)
        self.panel_view.grid_rowconfigure(0,weight=1)
        self.panel_view.grid(column=1,row=0,sticky='NWES')

        self.panel_sidebar = tk.Frame(self.mainframe, bg='lightgray')
        self.panel_sidebar.grid_columnconfigure(0,weight=1)
        self.panel_sidebar.grid_rowconfigure(0,weight=1)
        self.panel_sidebar.grid_rowconfigure((1,2),weight=6)
        self.panel_sidebar.grid(column=0, row=0, sticky='NWES')

        # Subpabels of sidebar
        self.panel_title = tk.Frame(self.panel_sidebar, bg='darkslategray')
        self.panel_title.grid_columnconfigure(0,weight=1)
        self.panel_title.grid_rowconfigure(0,weight=1)
        self.panel_title.grid(column=0,row=0, sticky='NWES')

        self.panel_params = tk.Frame(self.panel_sidebar, borderwidth=2, relief='ridge', padx=5, pady=5)
        self.panel_params.grid_columnconfigure(0, weight=2)
        self.panel_params.grid_columnconfigure((1,2), weight=1)
        self.panel_params.grid_rowconfigure((1,2,3,4),weight=1)
        self.panel_params.grid(column=0, row=1,sticky='NWES')

        self.panel_sim = tk.Frame(self.panel_sidebar, borderwidth=2, relief='ridge', padx=5, pady=5)
        self.panel_sim.grid_columnconfigure(0, weight=2)
        self.panel_sim.grid_columnconfigure((1,2), weight=1)
        self.panel_sim.grid_rowconfigure(4,weight=1)
        self.panel_sim.grid(column=0, row=2,sticky='NWES')

    def init_sidebar_components(self):
        # Title
        self.label_title = ttk.Label(self.panel_title, anchor='center', text="DRONE SWARM BOUNDARIES \n&\n POSE ESTIMATION", justify='center',
                                     font=font.Font(name='Helvetica', weight='bold', size=16), foreground='white', background=self.panel_title['bg'])
        self.label_title.grid(column=0,row=0, sticky='NEWS')

        # Drone number
        self.label_drone_nb = ttk.Label(self.panel_params, anchor='w', text="# drones: ")
        self.label_drone_nb.grid(column=0,row=0, sticky='NEWS')
        self.spinner_drone_nb = ttk.Spinbox(self.panel_params, increment=1,from_=0, to=50)
        self.spinner_drone_nb.grid(row=0,column=1,sticky='w', padx=5)

        # Swarm spread
        self.label_swarm_spread = ttk.Label(self.panel_params, anchor='w', text="Swarm spread: ")
        self.label_swarm_spread.grid(column=0,row=1, sticky='NEWS')
        self.slider_spread = ttk.Scale(self.panel_params, from_=0,to=20, value=10, orient='horizontal')
        self.slider_spread.grid(row=1, column=1, sticky='WE', padx=5)

        # Noise
        

if __name__ == "__main__":
    root = tk.Tk()
    #if len(sys.argv) > 1:
    #    # Set width and height
    #    root.geometry('{0}x{1}+0+0'.format(sys.argv[1], sys.argv[2]))
    root.title("Swarm boundaries simulation")
    root.geometry('{0}x{1}+0+0'.format(w, h))
    app = myApp(root)
    root.mainloop()
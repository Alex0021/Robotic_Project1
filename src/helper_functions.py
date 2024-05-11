import numpy as np
from mpl_toolkits.mplot3d import proj3d

class scatter():
    """
    A class to create a scatter plot with markers sizes in data units.

    (Solution found on StackOverflow:
    https://stackoverflow.com/questions/48172928/scale-matplotlib-pyplot-axes-scatter-markersize-by-x-scale/48174228#48174228)
    """
    def __init__(self,x,y,ax,radius=1,**kwargs):
        self.n = len(x)
        self.ax = ax
        self.ax.figure.canvas.draw()
        self.size_data=radius
        self.size = radius
        self.sc = ax.scatter(x,y,s=self.size,**kwargs)
        self._resize()
        self.cid = ax.figure.canvas.mpl_connect('draw_event', self._resize)

    def set_data(self,data_xy):
        self.sc.set_offsets(data_xy)

    def _resize(self,event=None):
        ppd=72./self.ax.figure.dpi
        trans = self.ax.transData.transform
        s =  np.mean((trans((2*self.size_data,2*self.size_data))-trans((0,0)))*ppd)
        self.sc.set_sizes(s**2*np.ones(self.n))
        self.size = s
        #self._redraw_later()

    def _redraw_later(self):
        self.timer = self.ax.figure.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.ax.figure.canvas.draw_idle())
        self.timer.start()

class scatter3D():
    def __init__(self,x,y,z,ax3d,radius=1,**kwargs):
        self.n = len(x)
        self.ax = ax3d
        self.ax.figure.canvas.draw()
        self.size_data=radius
        self.size = radius
        self.sc = ax3d.scatter(x,y,z,s=self.size,**kwargs)
        self._resize()
        self.cid = ax3d.figure.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self,event=None):
        ppd=72./self.ax.figure.dpi
        trans = self.ax.transData.transform
        x0, y0, _ = proj3d.proj_transform(0,0,0,self.ax.get_proj())
        x1, y1, _ = proj3d.proj_transform(1,1,1,self.ax.get_proj())
        #s = np.mean((trans((x1,y1)) - trans((x0,y0)))*ppd)
        s =  np.mean((trans((2*self.size_data*x1,2*self.size_data*y1))-trans((x0,y0)))*ppd)
        self.sc.set_sizes(s**2*np.ones(self.n))
        self.size = s
        self._redraw_later()

    def set_data(self,x,y,z):
        self.sc.set_offsets(np.c_[x,y])
        self.sc._offsets3d = np.c_[x,y,z]
        self._resize()

    def _redraw_later(self):
        self.timer = self.ax.figure.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.ax.figure.canvas.draw_idle())
        self.timer.start()
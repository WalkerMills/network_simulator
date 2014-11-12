import tkinter


class NetworkInput(tkinter.Frame):

    def __init__(self, master, width_=600, height_=400):
        super(NetworkInput, self).__init__(master)
        self.canvas = tkinter.Canvas(self, width=width_, height=height_)
        self.canvas.pack(fill='both', expand='1')

        self.dim = 10
        self.start = None

        self.hosts = 0
        self.routers = 0
        self.links = list()
        self.flows = list()

        self.master.bind('h', self.draw_host)
        self.master.bind('r', self.draw_router)
        self.canvas.bind('<Button-1>', self.draw_link)

    def _draw_component(self, x, y, color, tag):
        self.canvas.create_rectangle(x, y, x + self.dim, y + self.dim,
            fill=color, tags=tag)

    def draw_host(self, event):
        self._draw_component(event.x, event.y, '#66FF33',
                             'h{}'.format(self.hosts))
        self.hosts += 1

    def draw_router(self, event):
        self._draw_component(event.x, event.y, '#3366FF',
                             'r{}'.format(self.routers))
        self.routers += 1

    def draw_link(self, event):
        if self.start is None:
            self.start = event.x, event.y
            return

        self.canvas.create_line(self.start[0], self.start[1], event.x, event.y,
                                fill='black')
        self.start = None

master = tkinter.Tk()
canvas = NetworkInput(master)
canvas.pack(fill='both', expand='1')
tkinter.mainloop()

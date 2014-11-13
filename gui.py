import logging
import tkinter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NetworkInput(tkinter.Frame):

    def __init__(self, master, width_=600, height_=400):
        # Initialize GUI
        super(NetworkInput, self).__init__(master)
        self.canvas = tkinter.Canvas(self, width=width_, height=height_)
        self.canvas.pack(fill='both', expand='1')

        # Side length of hosts & routers in pixels
        self.dim = 10
        # Starting coordinates for a link
        self.start = None

        # Current number of hosts
        self.hosts = 0
        # Current number of routers
        self.routers = 0
        # Adjacency list (implicitly) representing links
        self.links = list()
        self.flows = list()

        # Bind 'h' key to host creation
        self.master.bind('h', self.draw_host)
        # Bind 'r' key to router creation
        self.master.bind('r', self.draw_router)
        # Bind left mouse button to link creation
        self.canvas.bind('<Button-1>', self.draw_link)

    def _draw_component(self, x, y, color, tag):
        """Draw a rectangular component."""
        self.canvas.create_rectangle(x, y, x + self.dim, y + self.dim,
            fill=color, tags=tag)

    def draw_host(self, event):
        """Draw a host."""
        self._draw_component(event.x, event.y, '#66FF33',
                             'h{}'.format(self.hosts))
        # Increment host count
        self.hosts += 1

    def draw_router(self, event):
        """Draw a router."""
        self._draw_component(event.x, event.y, '#3366FF',
                             'r{}'.format(self.routers))
        # Increment router count
        self.routers += 1

    def draw_link(self, event):
        """Draw a link."""
        # If there is no link in progress, a click represents a start point
        if self.start is None:
            self.start = event.x, event.y
            return

        link = list()
        for point in self.start, (event.x, event.y):
            # Find all nearby items
            nearby = self.canvas.find_overlapping(
                point[0] - self.dim / 2, point[1] - self.dim / 2,
                point[0] + self.dim / 2, point[1] + self.dim / 2)

            for item in nearby:
                # Retrieve the item's tags
                t = self.canvas.gettags(item)
                # If the item is not a link
                if not t[0].startswith('l'):
                    # This item is one of our link endpoints
                    link.append(t[0])
                    break

        logger.info('added link between {} and {}'.format(*link))
        link = ','.join(link)

        # If we have two valid endpoints
        if len(link) > 2:
            # Create a link
            self.canvas.create_line(self.start[0], self.start[1], event.x,
                                    event.y, fill='black', tags='l,' + link)
            self.links.append(link)
        # Reset start coordinates
        self.start = None

def draw():
    master = tkinter.Tk()
    canvas = NetworkInput(master)
    canvas.pack(fill='both', expand='1')
    tkinter.mainloop()

if __name__ == "__main__":
    draw()

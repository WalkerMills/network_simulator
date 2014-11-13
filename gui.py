import abc
import logging
import tkinter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dialog(tkinter.Toplevel, metaclass=abc.ABCMeta):
    """Abstract base class for constructing dialog boxes.

    This base class handles widget creation/destruction, and implements
    basic button functionality (Ok & Cancel buttons).  The dialog box
    content must be defined by subclasses by implementing a body method.
    Likewise, data validation is handled by the (abstract) validate
    method, and the actual processing/handling of input is done by the
    (abstract) apply method.

    Source:
        http://effbot.org/tkinterbook/tkinter-dialog-windows.htm
    """

    def __init__(self, parent, title=None):
        # Initialize top level widget
        super(Dialog, self).__init__(parent)

        # Set the parent widget
        self.parent = parent
        # Set the dialog title, if given
        if title is not None:
            self.title = title
        # Initialize return value to None
        self.result = None

        # Mark this dialog box as transient
        self.transient(parent)

        # Initialize dialog box content
        body = tkinter.Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)
        self.buttons()

        # Wait until the dialog box is visible
        self.wait_visibility()
        # Grab focus
        self.grab_set()
        if not self.initial_focus:
            self.initial_focus = self
        # Bind window closure to cancel handler
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        # Set dialog box geometry
        self.geometry("+{}+{}".format(parent.winfo_rootx() + 50,
                                      parent.winfo_rooty() + 50))
        # Set focus to dialog widget body
        self.initial_focus.focus_set()
        # Enter local event loop
        self.wait_window(self)

    @abc.abstractmethod
    def body(self, master):
        """Create dialog box body."""
        pass

    def buttons(self):
        """Create dialog box buttons."""
        # Initialize button container
        button_box = tkinter.Frame(self)
        # Create 'Ok' button
        widget = tkinter.Button(button_box, text='Ok', width=10,
                                command=self.ok, default=tkinter.ACTIVE)
        widget.pack(side=tkinter.LEFT, padx=5, pady=5)
        # Create 'Cancel' button
        widget = tkinter.Button(button_box, text='Cancel', width=10,
                                command=self.cancel)
        widget.pack(side=tkinter.LEFT, padx=5, pady=5)
        # Build button container
        button_box.pack()

        # Bind enter key to the 'Ok' button
        self.bind('<Return>', self.ok)
        # Bind esc key to the 'Cancel' button
        self.bind('<Escape>', self.cancel)

    def ok(self, event=None):
        """Define the action of the \'Ok\' button."""
        # If we got invalid data
        if not self.validate():
            # Reset focus
            self.initial_focus.focus_set()
            return

        # Remove this dialog box, as far as the window manager is concerned
        self.withdraw()
        # Trigger all idle callbacks
        self.update_idletasks()
        # Process inputted data
        self.apply()
        # Destroy this widget & clean up
        self.cancel()

    def cancel(self, event=None):
        """Define the action of the \'Cancel\' button."""
        # Return focus to the parent widget
        self.parent.focus_set()
        # Destroy this widget
        self.destroy()

    @abc.abstractmethod
    def validate(self):
        """Validate the data entered in the dialog box."""
        pass

    @abc.abstractmethod
    def apply(self):
        """Process the data entered in the dialog box."""
        pass


class NetworkInput(tkinter.Frame):
    """This class implements a GUI to draw a network configuration."""


    class LinkDialog(Dialog):
        """Dialog for specifying link parameters."""

        attributes = ['Capacity', 'Delay', 'Buffer size']

        def _get_entry(self, i):
            """Get the (integer) value of the ith entry field."""
            return int(getattr(self, 'e{}'.format(i)).get())

        def body(self, master):
            for i, att in enumerate(self.attributes):
                # Make a label for this attribute
                tkinter.Label(master, text=att + ':').grid(row=i)
                # Make an entry field for this attribute
                entry = 'e{}'.format(i)
                setattr(self, entry, tkinter.Entry(master))
                # Position this entry field
                getattr(self, entry).grid(row=i, column=1)

        def validate(self):
            # Check that all values are >= 0
            return all(map(lambda i: self._get_entry(i) >= 0,
                           range(len(self.attributes))))

        def apply(self):
            # Store the entered parameters in this dialog's result
            self.result = [self._get_entry(i) for i in 
                           range(len(self.attributes))]


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

        # If we have two valid endpoints
        if len(link) == 2:
            logger.info('creating new link between {} and {}'.format(*link))
            link = ','.join(link)
            # Get link parameters
            dialog = NetworkInput.LinkDialog(self)
            # Draw a link in the GUI
            self.canvas.create_line(self.start[0], self.start[1], event.x,
                                    event.y, fill='black', tags='l,' + link)
            # Update the list of links
            self.links.append([link] + dialog.result)
        else:
            logger.info('link creation failed; endpoints not valid')
        # Reset start coordinates
        self.start = None

def draw():
    """Draw the network configuration GUI."""
    # Initialize GUI
    master = tkinter.Tk()
    canvas = NetworkInput(master)
    canvas.pack(fill='both', expand='1')
    # Run GUI
    tkinter.mainloop()

if __name__ == "__main__":
    draw()

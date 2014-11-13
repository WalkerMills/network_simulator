import abc
import logging
import operator
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
        # Create "Ok" button
        widget = tkinter.Button(button_box, text="Ok", width=10,
                                command=self.ok, default=tkinter.ACTIVE)
        widget.pack(side=tkinter.LEFT, padx=5, pady=5)
        # Create "Cancel" button
        widget = tkinter.Button(button_box, text="Cancel", width=10,
                                command=self.cancel)
        widget.pack(side=tkinter.LEFT, padx=5, pady=5)
        # Build button container
        button_box.pack()

        # Bind enter key to the "Ok" button
        self.bind("<Return>", self.ok)
        # Bind esc key to the "Cancel" button
        self.bind("<Escape>", self.cancel)

    def ok(self, event=None):
        """Define the action of the \"Ok\" button."""
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
        """Define the action of the \"Cancel\" button."""
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


class InputDialog(Dialog):
    """Dialog for entering field data."""

    fields = list()

    def _get_entry(self, i):
        """Get the (integer) value of the ith entry field."""
        return int(getattr(self, "e{}".format(i)).get())

    def body(self, master):
        """Create a labeled entry box for each field."""
        for i, att in enumerate(self.fields):
            # Make a label for this attribute
            tkinter.Label(master, text=att + ":").grid(row=i)
            # Make an entry field for this attribute
            entry = "e{}".format(i)
            setattr(self, entry, tkinter.Entry(master))
            # Position this entry field
            getattr(self, entry).grid(row=i, column=1)

    def apply(self):
        """Store the values entered in the entry fields."""
        # Store the entered parameters in this dialog's result
        self.result = [self._get_entry(i) for i in range(len(self.fields))]



class NetworkInput(tkinter.Frame):
    """This class implements a GUI to draw a network configuration."""

    class LinkDialog(InputDialog):
        """Dialog for specifying link parameters."""

        fields = ["Capacity', 'Delay', 'Buffer size"]

        def validate(self):
            """Check that all values entered are >= 0"""
            return all(map(lambda i: self._get_entry(i) >= 0,
                           range(len(self.fields))))

    class FlowDialog(InputDialog):
        """Dialog for specifying flow parameters"""

        fields = ["Window', 'Timeout', 'Data"]

        def validate(self):
            """Check that all values entered are >= 0"""
            return all(map(lambda i: self._get_entry(i) >= 0,
                           range(len(self.fields))))

    def __init__(self, master, width_=600, height_=400):
        # Initialize GUI
        super(NetworkInput, self).__init__(master)
        self.canvas = tkinter.Canvas(self, width=width_, height=height_)
        self.canvas.pack(fill="both', expand='1")

        # Side length of hosts & routers in pixels
        self._dim = 10
        # Starting coordinates for a link
        self._start = None
        # Initial host for a flow
        self._src = None
        # Current number of hosts
        self._hosts = 0
        # Current number of routers
        self._routers = 0
        # Adjacency list (implicitly) representing links
        self.links = list()
        # List of flows in the network
        self.flows = list()

        # Bind "h" key to host creation
        self.master.bind("h", self.draw_host)
        # Bind "r" key to router creation
        self.master.bind("r", self.draw_router)
        # Bind left mouse button to link creation
        self.canvas.bind("<Button-1>", self.draw_link)
        # Bind right mouse button to flow creation
        self.canvas.bind("<Button-3>", self.make_flow)

    def _draw_component(self, x, y, color, tag):
        """Draw a rectangular component."""
        self.canvas.create_rectangle(x, y, x + self._dim, y + self._dim,
            fill=color, tags=tag)

    def _find_item(self, x, y, tag, invert=False):
        """Find an item (prefix) matching a given tag.

        This method finds all items near the coordinates given, and
        returns the first tag of the closest item whose first tag prefix
        matches the tag given.  If the invert parameter is True, then
        the matching is inverted, and the first item whose first tag
        doesn't prefix match tag is returned.  If no such item is found,
        this method returns None.
        """
        # If invert is true, check for an item that doesn't match <tag>
        truth = operator.not_ if invert else operator.truth
        # Find nearby items
        nearby = self.canvas.find_overlapping(
            x - self._dim / 2, y - self._dim / 2,
            x + self._dim / 2, y + self._dim / 2)
        for item in nearby:
            # Retrieve the item's tags
            tags = self.canvas.gettags(item)
            # If the item matches correctly to the given tag
            if truth(tags[0].startswith(tag)):
                # This item is an endpoint
                return tags[0]
        # Return None if no valid endpoint was found
        return None

    def draw_host(self, event):
        """Draw a host."""
        self._draw_component(event.x, event.y, "#66FF33",
                             "h{}".format(self._hosts))
        # Increment host count
        self._hosts += 1

    def draw_router(self, event):
        """Draw a router."""
        self._draw_component(event.x, event.y, "#3366FF",
                             "r{}".format(self._routers))
        # Increment router count
        self._routers += 1

    def draw_link(self, event):
        """Draw a link."""
        # If there is no link in progress
        if self._start is None:
            # The click event is the start point of a link
            self._start = event.x, event.y
            return

        link = list()
        for point in self._start, (event.x, event.y):
            endpoint = self._find_item(point[0], point[1], "l", invert=True)
            if endpoint is not None:
                link.append(endpoint)

        # If we have two valid endpoints
        if len(link) == 2:
            # Get link parameters
            dialog = NetworkInput.LinkDialog(self)
            # If we got valid link parameters
            if dialog.result is not None:
                logger.info("creating new link from {} to {}".format(*link))
                link = ",".join(link)
                # Draw a link in the GUI
                self.canvas.create_line(self._start[0], self._start[1], 
                                        event.x, event.y, fill="black", 
                                        tags="l," + link)
                # Update the list of links
                self.links.append([link] + dialog.result)
            else:
                logger.info("link creation failed; invalid link parameters")
        else:
            logger.info("link creation failed; endpoints not valid")
        # Reset start coordinates
        self._start = None

    def make_flow(self, event):
        """Make a flow from one host to another."""
        # Find the closest host
        endpoint = self._find_item(event.x, event.y, "h")

        # If we found a host
        if endpoint is not None:
            # If there is no flow in progress
            if self._src is None:
                # The endpoint we found is the source of a flow
                self._src = endpoint
                return

            # Get flow parameters
            dialog = NetworkInput.FlowDialog(self)
            # If we got valid flow parameters
            if dialog.result is not None:
                logger.info("creating new flow from {} to {}".format(self._src,
                                                                     endpoint))
                # Update the list of flows
                self._flows.append(
                    [self._src + "," + endpoint] + dialog.result)
        else:
            logger.info("invalid flow endpoint")
        # Reset flow source
        self._src = None


def draw():
    """Draw the network configuration GUI."""
    # Initialize GUI
    master = tkinter.Tk()
    canvas = NetworkInput(master)
    canvas.pack(fill="both', expand='1")
    # Run GUI
    tkinter.mainloop()
    return canvas.links, canvas.flows

if __name__ == "__main__":
    draw()

"""
.. module:: gui
    :platform: Unix
    :synopsis: This module implements a GUI for drawing a network.
"""

import abc
import logging
import operator
import tkinter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Dialog(tkinter.Toplevel, metaclass=abc.ABCMeta):
    """Abstract base class for constructing dialog boxes.

    This base class handles widget creation/destruction, and implements
    basic button functionality (Ok & Cancel buttons).  The dialog box
    content must be defined by subclasses by implementing a body method.
    Likewise, data validation is handled by the (abstract) validate
    method, and the actual processing/handling of input is done by the
    (abstract) apply method.

    :param parent: parent of this widget
    :type parent: tkinter.Widget
    :param str title: dialog box title

    `Source <http://effbot.org/tkinterbook/tkinter-dialog-windows.htm>`_
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
        """Create dialog box body.

        :param master: parent of this widget
        :type master: tkinter.Frame

        :return: the widget with initial focus
        :rtype: tkinter.Widget
        """
        pass

    def buttons(self):
        """Create dialog box buttons.

        :return: None
        """
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
        """Define the action of the \"Ok\" button.

        :return: None
        """
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
        """Define the action of the \"Cancel\" button.

        :return: None
        """
        # Return focus to the parent widget
        self.parent.focus_set()
        # Destroy this widget
        self.destroy()

    @abc.abstractmethod
    def validate(self):
        """Validate the data entered in the dialog box.

        :return: True iff the data is valid
        :rtype: bool
        """
        pass

    @abc.abstractmethod
    def apply(self):
        """Process the data entered in the dialog box.

        :return: None
        """
        pass


class InputDialog(Dialog):
    """Dialog for entering field data.

    :param parent: parent of this widget
    :type parent: tkinter.Widget
    :param str title: dialog box title
    """

    labels = list()

    def _get_entry(self, i):
        """Get the (integer) value of the ith entry field."""
        return int(getattr(self, "f{}".format(i)).get())

    def body(self, master):
        """Create a labeled entry box for each label.

        :param master: parent of this widget
        :type master: tkinter.Frame

        :return: the entry field with initial focus, or None
        :rtype: tkinter.Widget
        """
        for i, label in enumerate(self.labels):
            # Make a label in the label box for this field
            tkinter.Label(master, text=label + ":").grid(row=i)
            # Make an entry field for this attribute
            entry = "f{}".format(i)
            setattr(self, entry, tkinter.Entry(master))
            # Position this entry field
            getattr(self, entry).grid(row=i, column=1)

        if hasattr(self, 'f0'):
            return self.f0

    def apply(self):
        """Store the values entered in the entry labels.

        :return: None
        """
        # Store the entered parameters in this dialog's result
        self.result = [self._get_entry(i) for i in range(len(self.labels))]


class FlowDialog(InputDialog):
    """Dialog for specifying flow parameters.

    :param parent: parent of this widget
    :type parent: tkinter.Widget
    :param str title: dialog box title
    """

    labels = ["Data", "Window", "Timeout", "Initial delay"]
    """Flow positional parameters (label names)."""

    def validate(self):
        """Check that all values entered are >= 0.

        :return: True iff the data is valid
        :rtype: bool
        """
        return all(map(lambda i: self._get_entry(i) >= 0,
                       range(len(self.labels))))


class LinkDialog(InputDialog):
    """Dialog for specifying link parameters.

    :param parent: parent of this widget
    :type parent: tkinter.Widget
    :param str title: dialog box title
    """

    labels = ["Capacity", "Buffer size", "Delay"]
    """Link positional parameters (label names)."""

    def validate(self):
        """Check that all values entered are >= 0.

        :return: True iff the data is valid
        :rtype: bool
        """
        return all(map(lambda i: self._get_entry(i) >= 0,
                       range(len(self.labels))))


class NetworkInput(tkinter.Frame):
    """This class implements a GUI to draw a network configuration.

    :param master: Parent of this widget
    :type master: tkinter.Tk
    :param width_: Width of the GUI window
    :type width_: int
    :param height_: Height of the GUI window
    :type height_: int
    """

    def __init__(self, master, width_=600, height_=400):
        # Initialize GUI
        super(NetworkInput, self).__init__(master)
        self.canvas = tkinter.Canvas(self, width=width_, height=height_)
        self.canvas.pack(fill="both", expand="1")

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
        self._links = list()
        # List of flows in the network
        self._flows = list()

        # Bind "h" key to host creation
        self.master.bind("h", self.draw_host)
        # Bind "r" key to router creation
        self.master.bind("r", self.draw_router)
        # Bind left mouse button to link creation
        self.canvas.bind("<Button-1>", self.draw_link)
        # Bind right mouse button to flow creation
        self.canvas.bind("<Button-3>", self.make_flow)

    @property
    def links(self):
        """The links connected on this canvas.

        :return: an adjacency list specifying the network topology
        :rtype: list
        """
        return self._links

    @property
    def flows(self):
        """The flows defined on this canvas

        :return: a list of flows defined in this network
        :rtype: list
        """
        return self._flows

    def _draw_component(self, x, y, color, tag):
        """Draw a rectangular component."""
        self.canvas.create_rectangle(x, y, x + self._dim, y + self._dim,
            fill=color, tags=tag)

    def _find_item(self, x, y, tag, invert=False):
        """Find a nearby item (prefix) matching a given tag.

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
        """Draw a host.

        :param event: \"h\" keypress event
        :type event: tkinter.Event

        :return: None
        """
        self._draw_component(event.x, event.y, "#66FF33",
                             "h{}".format(self._hosts))
        # Increment host count
        self._hosts += 1

    def draw_router(self, event):
        """Draw a router.

        :param event: \"r\" keypress event
        :type event: tkinter.Event

        :return: None
        """
        self._draw_component(event.x, event.y, "#3366FF",
                             "r{}".format(self._routers))
        # Increment router count
        self._routers += 1

    def draw_link(self, event):
        """Draw a link.

        :param event: left mouse button click event
        :type event: tkinter.Event

        :return: None
        """
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
            dialog = LinkDialog(self, "Link Parameters")
            # If we got valid link parameters
            if dialog.result is not None:
                logger.info("creating new link from {} to {}".format(*link))
                # Draw a link in the GUI
                self.canvas.create_line(self._start[0], self._start[1], 
                                        event.x, event.y, fill="black", 
                                        tags="l," + ','.join(link))
                # Update the list of links
                self._links.append((tuple(link), tuple(dialog.result)))
            else:
                logger.info("link creation failed; invalid link parameters")
        else:
            logger.info("link creation failed; endpoints not valid")
        # Reset start coordinates
        self._start = None

    def make_flow(self, event):
        """Make a flow from one host to another.

        :param event: right mouse button click event
        :type event: tkinter.Event

        :return: None
        """
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
            dialog = FlowDialog(self, "Flow Parameters")
            # If we got valid flow parameters
            if dialog.result is not None:
                logger.info("creating new flow from {} to {}".format(self._src,
                                                                     endpoint))
                # Update the list of flows
                self._flows.append(((self._src, endpoint), 
                                    tuple(dialog.result)))
        else:
            logger.info("invalid flow endpoint")
        # Reset flow source
        self._src = None

def draw():
    """Draw the network configuration GUI.

    :return: returns drawn network as (adjacency list, flows)
    :rtype: (list, list)
    """
    # Initialize GUI
    master = tkinter.Tk()
    canvas = NetworkInput(master)
    canvas.pack(fill="both", expand="1")
    # Run GUI
    tkinter.mainloop()
    return canvas.links, canvas.flows

if __name__ == "__main__":
    draw()

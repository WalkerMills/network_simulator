"""
.. module:: gui
    :platform: Unix
    :synopsis: This module implements a GUI for drawing a network.
"""

import logging
import operator
import tkinter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Dialog(tkinter.Toplevel):
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
        content = tkinter.Frame(self)
        self.body(content)
        content.pack(padx=5, pady=5)
        self.buttons()

        # Wait until the dialog box is visible
        self.wait_visibility()
        # Grab focus
        self.grab_set()
        # Bind window closure to cancel handler
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        # Set dialog box geometry
        self.geometry("+{}+{}".format(parent.winfo_rootx() + 50,
                                      parent.winfo_rooty() + 50))
        # Set focus to dialog widget body
        self.focus_set()
        # Enter local event loop
        self.wait_window(self)

    def body(self, master):
        """Create dialog box body.

        Should be overridden by subclasses.

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
        # If we got invalid data, don't destroy the widget
        if not self.validate():
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

    def validate(self):
        """Validate the data entered in the dialog box.

        :return: True iff the data is valid
        :rtype: bool
        """
        pass

    def apply(self):
        """Process the data entered in the dialog box.

        :return: None
        """
        pass


class InputDialog(Dialog):
    """Base dialog box class for entering field data.

    :param parent: parent of this widget
    :type parent: tkinter.Widget
    :param str title: dialog box title
    """

    labels = list()
    """Entry field labels."""

    types = list()
    """Input types."""

    def _get_entry(self, i, type_=str):
        """Get the value of the ith entry field, and cast it."""
        return type_(getattr(self, "f{}".format(i)).get())

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
        self.result = tuple(
            self._get_entry(i, self.types[i]) for i in range(len(self.labels)))


class FlowDialog(InputDialog):
    """Dialog box for specifying flow parameters.

    The last entry in this dialog box is the TCP specification.  Since
    all TCP algorithms take a window size & timeout, they appear as
    entries by default; any additional arguments should be specified in
    the last entry box, as a comma-separated list of values following a
    valid TCP specifier.  See :data:`process.Flow.allowed_tcp` for a
    list of TCP specifiers, and the corresponding classes.

    :param parent: parent of this widget
    :type parent: tkinter.Widget
    :param str title: dialog box title
    """

    labels = ["Data", "Initial delay", "Window", "Timeout", "TCP"]
    """Flow positional parameters (label names)."""

    types = [int, int, int, int, {"FAST": [int, float], "Reno": []}]
    """Input types."""

    def validate(self):
        """Check that all values entered are valid.

        All values must be >= 0, and the TCP specifier must be one of
        those supported by :class:`process.Flow`.

        :return: True iff the data is valid
        :rtype: bool
        """
        # Entry value type generator
        type_gen = (t for t in self.types)
        # Check that all but the TCP parameters are valid (>= 0)
        valid = all(map(lambda i: self._get_entry(i, next(type_gen)) >= 0,
                        range(len(self.labels) - 1)))
        # Get the TCP type dict
        tcp_types = next(type_gen)
        # Extract the entered TCP values
        tcp, *tcp_param = self._get_entry(len(self.labels) - 1).split(',')
        # Check that the TCP specifier is valid
        valid = valid and tcp in tcp_types.keys()
        if valid:
            # Get the appropriate list of parameter types
            tcp_types = tcp_types[tcp]
            # Check that all the TCP parameters are valid
            valid = valid and all(map(lambda p: p[0](p[1]) >= 0,
                                      zip(tcp_types, tcp_param)))
        return valid

    def apply(self):
        """Store the values entered in the entry labels.

        Values are cast according to FlowDialog.types; the last value maps
        allowed TCP specifiers to the types they expect.

        :return: None
        """
        # Entry field value generator
        values = (self._get_entry(i) for i in range(len(self.labels)))
        # Entry value type generator
        type_gen = (t for t in self.types)
        # Get the flow parameters
        flow = tuple(next(type_gen)(next(values)) for i in range(2))
        # Get the flow's data size
        window = next(type_gen)(next(values))
        # Get the flow's initial delay
        timeout = next(type_gen)(next(values))
        # Extract the TCP specifier & parameters from the last argument
        tcp, *raw_param = next(values).split(',')
        # Get the TCP type hash table
        tcp_types = next(type_gen)
        # TCP parameter type generator
        tcp_types = (t for t in tcp_types[tcp])
        # Typed TCP parameters
        tcp_param = list()
        # Cast the TCP parameters
        for param, t in zip(raw_param, tcp_types):
            tcp_param.append(t(param))
        # Set the results
        self.result = (flow, (tcp, [window, timeout] + tcp_param))


class LinkDialog(InputDialog):
    """Dialog box for specifying link parameters.

    :param parent: parent of this widget
    :type parent: tkinter.Widget
    :param str title: dialog box title
    """

    labels = ["Capacity", "Buffer size", "Delay"]
    """Link positional parameters (label names)."""

    types = [int, int, int]
    """Link parameter types."""

    def validate(self):
        """Check that all values entered are >= 0.

        :return: True iff the data is valid
        :rtype: bool
        """
        return all(map(lambda i: self._get_entry(i, self.types[i]) >= 0,
                       range(len(self.labels))))


class NetworkInput(tkinter.Frame):
    """This class implements a GUI to draw a network configuration.

    In the GUI, pressing \"h\" places a host under the cursor, pressing
    \"r\" places a router under the cursor, left-clicking one component,
    then a different component, draws a link between the two, and right-
    clicking one host, then a different host, draws a flow between those
    hosts.  After two valid endpoints have been given for a link or flow,
    a dialog box will appear and prompt the user for link/flow parameters.
    See :class:`LinkDialog` or :class:`FlowDialog` for details on what
    values these dialog boxes expect.

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
        """The flows defined on this canvas.

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
            if endpoint is not None and endpoint not in link:
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
            # Return if the endpoint is already the source
            if endpoint == self._src:
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

    See :class:`test.Network` for details on the output format.

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

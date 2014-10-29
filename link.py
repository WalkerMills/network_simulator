import queue
import simpy


class Link:
    """Simulator object representing a link."""

    def __init__(self, env, capacity, delay, rate):
        self.env = env

        # TODO: check capacity, delay, & rate for valid values

        # Maximum queue size
        self._capacity = capacity
        # Link delay
        self._delay = delay
        # Link rate
        self._rate = rate

        self.items = queue.Queue(self._capacity)

    @property
    def capacity(self):
        """The maximum capacity of the link queue."""
        return self._capacity

    @property
    def delay(self):
        """The link delay."""
        return self._delay

    @property
    def rate(self):
        """The maximum bitrate of the link."""
        return self._rate

    def put(self, event):
        """Add a packet the link queue, or drop it on queue overflow."""
        pass

    def send(self, event):
        """Send a packet through the link."""
        pass

   # TODO: implement dynamic cost method, rename this one
    def cost(self):
        """Calculate the cost of this link.

        Cost is proportional to link capacity & rate, and inversely
        proportional to delay.

        """
        return self._capacity * self._rate / self._delay

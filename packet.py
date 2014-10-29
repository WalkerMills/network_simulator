import simpy


class Packet(object):
    """This class represents a packet in our simulation."""

    def __init__(self, src, dest, flow_id, payload):
        # Packet source address
        self._src = src
        # Packet destination address
        self._dest = dest
        # Flow ID on the source host
        self._flow = flow_id
        # Packet data
        self._data = payload

        # Simulated packet size
        self._size = 1024

    @property
    def src(self):
        """Return the packet's source address."""
        return self._src

    @property
    def dest(self):
        """Return the packet's destination address."""
        return self._dest

    @property
    def size(self):
        """Return the packet size in bytes."""
        return self._size

    @property
    def flow(self):
        """Return the ID of the sender (flow) on the source host."""
        return self._flow

    @property
    def data(self):
        """Return this packet's payload of data."""
        return self._data


class ACK(Packet):
    """This class represents an acknowedgement packet."""

    def __init__(self, src, dest, flow, payload):
        super(ACK, self).__init__(src, dest, flow, payload)
        self._size = 64


class ReceivePacket(simpy.events.Event):
    """Simulator event representing packet arrival.

    This event takes a resource as a parameter, and represents the 
    arrival of a packet at that resource.  It also triggers a packet
    transmission for each packet received.  This event may also be used
    as a context manager.
    """

    def __init__(self, resource, packet):
        # Initialize event
        super(ReceivePacket, self).__init__(resource._env)
        self.resource = resource
        self.packet = packet

        # Record this event's active process
        self.proc = self.env.active_process
        # Add this event to the packet queue
        resource._packets.append(self)
        # Send a packet, since we have enqueued a new packet
        self.resource._trigger_transmit()

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        # If the event was interrupted, dequeue it
        if not self.triggered:
            self.resource._packets.remove(self)

    cancel = __exit__

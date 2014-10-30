import queue
import simpy


class Packet(object):
    """This class represents a packet in our simulation."""

    def __init__(self, src, dest, fid, pid, payload):
        # Packet source address
        self._src = src
        # Packet destination address
        self._dest = dest
        # Flow ID on the source host
        self._flow = fid
        # Packet data
        self._data = payload

        # Simulated packet size
        self._size = 1024
        # Packet ID
        self._id = pid

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

    def id(self):
        """Return the ID of this packet."""
        return self._id

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

        # Add this event to the packet queue
        resource._packets.append(self)
        # Send a packet, since we enqueued a new packet
        self.resource._trigger_transmit()

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        # If the event was interrupted, dequeue it
        if not self.triggered:
            self.resource._packets.remove(self)

    cancel = __exit__


class HostResource(object):
    """Resource representing a host.
    
    This is a host implemented as a unidirectional resource.  It receives
    packets, but does not accept packet transmission requests.  Every
    packet arrival triggers a packet transmission, at which time the
    next packet in the queue is popped.  Inbound data packets trigger
    an automatic acknowledgement, and inbound ACK packets are sent to
    the appropriate flow.  Outbound packets are transmitted via the
    outbound link, if it exists.

    """

    def __init__(self, env, addr):
        self._env = env
        # host address
        self._addr = addr
        # Queue to hold packets to transmit
        self._packets = list()
        # Outbound link
        self._link = None
        # Active flows
        self._flows = list()

        simpy.core.BoundClass.bind_early(self)

    receive = simpy.core.BoundClass(packet.ReceivePacket)

    @property
    def addr(self):
        """The address of this host."""
        return self._addr

    @property
    def link(self):
        """The outbound link connected to this host."""
        return self._link

    @link.setter
    def link(self, link):
        self._link = link

    @property
    def flows(self):
        """A list of flows active on this host."""
        return self._flows

    def register(self, flow):
        """Register a new flow on this host, and return the flow ID."""
        self._flows.append(flow)
        return len(self._flows) - 1

    def _transmit(self, packet):
        """transmit an outbound packet."""
        if packet.dest == self.addr:
            # TODO: send ACK back to source flow
            pass
        else:
            # TODO: send outbound packet through outbound link
            pass

    def _trigger_transmit(self):
        """Trigger outbound packet transmission."""
        event = self._packets.pop(0)

        # If a data packet has reached its destination, transmit an ACK
        # back to the source flow
        if event.packet.dest == self.addr && event.packet.size == 1024:
            ack = ACK(self.addr, event.packet.src, event.packet.flow
                      event.packet.id + 1)
            self._transmit(ack)
        # Otherwise, transmit the packet
        else:
            self._transmit(event.packet)
        event.succeed()


class RouterResource(object):
    """Resource representing a router.

    This class is a router implemented as a unidirectional resource.
    It receives packets, but does not accept packet transmission 
    requests.  Every packet arrival triggers a packet transmission, at
    which time the next packet in the queue is popped, routed, and sent
    through the appropriate link.
    """

    def __init__(self, env, dynamic=False):
        self._env = env
        self._dynamic = dynamic
        # Queue of packet events to process
        self._packets = list()
        # List of outbound links of the form (link cost, link)
        self._links = list()
        # Dictionary mapping outbound links to destination ranges
        self._table = dict()

        # Bind event constructors as methods
        simpy.core.BoundClass.bind_early(self)

    receive = simpy.core.BoundClass(ReceivePacket)

    @property
    def dynamic(self):
        return self._dynamic

    @dynamic.setter
    def dynamic(self, d):
        if d not in [True, False]:
            raise ValueError("Invalid dynamic flag value.")

        self._dynamic = d

    @property
    def links(self):
        """A list of tuples of the form (static cost, link)."""
        return self._links

    @links.setter
    def links(self, link):
        self._links.append((link.static_cost(), link))
        # TODO: add endpoints reachable through the link to self._table
        self._table[link] = [None]

    def disconnect(self, index):
        """Disconnect a link from this router, if it exists."""
        # Remove routing information for this link, if extant
        self._table.pop(self._links[index])
        # Remove the link
        self._links.remove(self._links[index])


    def route(self, dest, dynamic=False):
        in_range = lambda l: dest in self._table[l[1]]
        # TODO: dynamic routing updates link costs before choosing
        out = min(filter(in_range, self._links))

        return out

    def _transmit(self, packet):
        """Transmit an outbound packet."""
        # TODO: route packet & transmit it through a link
        pass

    def _trigger_transmit(self):
        """Trigger outbound packet transmission."""
        event = self._packets.pop(0)
        self._transmit(event.packet)
        event.succeed()


class LinkTransmit(simpy.events.Event):
    """Simulator event representing directional packet transmission.

    This event takes a resource as a parameter, and represents the
    transmission of a packet arcoss a full-duplex link, through the
    use of a limited size buffer.  Buffer overflows are dropped.
    """

    def __init__(self, link, direction, packet):
        # Initialize event
        super(LinkTransmit, self).__init__(link._env)
        self._link = link
        self._direction = direction
        self._packet = packet

        # Enqueue the packet for transmission, if the buffer isn't full
        if link.size - link._fill[direction] >= packet.size:
            # Increment buffer fill
            link._fill[direction] += packet.size
            # Enqueue packet
            link._packet_queues[direction].put_nowait(self)
            # Add a callback to update link rate on transmission completion
            self.callbacks.append(link._trigger_finish)


    def __enter__(self):
        # Transport a packet through the link
        self._link._trigger_transmit(self._direction)
        return self

    def __exit__(self, exception, value, traceback):
        # If the event was interrupted,
        # if not self.triggered:
        #     handle error
        pass

    @property
    def direction(self):
        """Link direction"""
        return self._direction

    @property
    def packet(self):
        """The packet to be transmitted"""
        return self._packet


class LinkResource(object):
    """Rousource representing a link.

    This class is a full-duplex link implemented as a resource.  Packet
    transmission is parametrized by direction, and each link has two
    allowed directions (0 or 1)
    """

    def __init__(self, env, capacity, delay, buf_size):
        self._env = env

        # Check capacity, delay, & rate for valid values
        if any(map(lambda opt: opt < 0, (capacity, delay, buf_size))):
            raise ValueError("capacity, delay, and buffer size must be >= 0")

        # Link rate (Bps)
        self._capacity = capacity
        # Link traffic (Bps)
        self._traffic = [0, 0]
        # Link delay (seconds)
        self._delay = delay
        # Buffer size (bytes)
        self._size = buf_size
        # Buffer fill (bytes)
        self._fill = [0, 0]

        # Buffers for each edge direction
        self._packet_queues = (queue.Queue(self._capacity),
                               queue.Queue(self._capacity))
        # Endpoints for each direction
        self._endpoints = [None, None]
        # Bind any classes into methods now
        simpy.core.BoundClass.bind_early(self)

    transport = simpy.core.BoundClass(LinkTransmit)
    """Transport packets across the link in a given direction."""

    @property
    def capacity(self):
        """The maximum bitrate of the link in Bps."""
        return self._capacity

    @property
    def delay(self):
        """The link delay in seconds."""
        return self._delay

    @property
    def size(self):
        """Maximum buffer capacity in bytes."""
        return self._size

    @property
    def endpoints(self):
        """A list of connected endpoints (=< 1 per direction)"""
        return self._endpoints

    @endpoints.setter
    def endpoints(self, direction, endpoint):
        self._endpoints[direction] = endpoint

    def disconnect(self, direction):
        """Disconnect a transmission endpoint."""
        self._endpoints[direction] = None        

    # TODO: implement dynamic cost method
    def static_cost(self):
        """Calculate the static cost of this link.

        Cost is inversely proportional to link capacity & rate, and 
        directly proportional to delay.

        """
        try:
            return self._delay / (self._capacity * self._rate)
        # If the link is down (0 capacity || 0 bps), return infinite cost
        except ZeroDivisionError:
            return float("inf")

    def _transmit(self, direction, packet):
        """Transmit a packet across the link in a given direction."""
        # Update link traffic
        self._traffic[direction] += packet.size
        # If the link is disconnected or full, drop the packet
        if self._endpoints[direction] == None or \
            self._capacity < self._traffic[direction]:
            return

        # TODO: send the packet to the resource at the appropriate endpoint

    def _trigger_transmit(self, direction):
        """Trigger packet transmission."""
        try:
            # Dequeue a packet
            event = self._packet_queues[direction].get_nowait()
            # Update buffer fill
            self._fill[direction] -= event.packet.size
        except queue.Empty:
            return

        # Transmit packet
        self._transmit(direction, event.packet)
        # Trigger event callbacks
        event.succeed()

    def _trigger_finish(self, event):
        """Update link rate after a packet is sent."""
        self._traffic[event.direction] -= event.packet.size

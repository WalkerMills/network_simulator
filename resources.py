import logging
import queue
import simpy
import simpy.util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Packet(object):
    """This class represents a packet in our simulation.

    :param src: source host
    :type src: :class:`process.Host`
    :param int dest: destination address
    :param int fid: source flow id
    :param int pid: packet id
    :param object payload: packet payload
    """

    # Simulated packet size (bits)
    size = 8192

    def __init__(self, src, dest, fid, pid, payload):
        # Packet source address
        self._src = src
        # Packet destination address
        self._dest = dest
        # Flow ID on the source host
        self._flow = fid
        # Packet ID
        self._id = pid
        # Packet data
        self._data = payload

    @property
    def src(self):
        """Return the packet's source address."""
        return self._src

    @property
    def dest(self):
        """Return the packet's destination address."""
        return self._dest

    @property
    def flow(self):
        """Return the ID of the sender (flow) on the source host."""
        return self._flow

    @property
    def id(self):
        """Return the ID of this packet."""
        return self._id

    @property
    def data(self):
        """Return this packet's payload of data."""
        return self._data

    def acknowledgement(self):
        """Generate an acknowledgement for this packet.

        :return: An acknowledgement packet matching this packet
        :rtype: :class:`ACK`
        """
        return ACK(self._dest, self._src, self._flow, self._id)


class ACK(Packet):
    """This class represents an acknowledgement packet.

    :param src: source host
    :type src: :class:`process.Host`
    :param int dest: destination address
    :param int fid: source flow id
    :param int pid: packet id
    """

    # Simulated acknowledgement packet size (bits)
    size = 512

    def __init__(self, src, dest, fid, pid):
        # Packet source address
        self._src = src
        # Packet destination address
        self._dest = dest
        # Flow ID on the source host
        self._flow = fid
        # ID of the packet which triggered this acknowledgement
        self._id = pid
        # Acknowledgement packets have no payload
        self._data = None


class ReceivePacket(simpy.events.Event):
    """Simulator event representing packet arrival.

    This event takes a resource as a parameter, and represents the 
    arrival of a packet at that resource.  It also triggers a packet
    transmission for each packet received.  This event may also be used
    as a context manager.

    :param object resource: 
        an object with a _packet attribute (container) and a _receive method
    :param packet: a packet to handle
    :type packet: :class:`Packet`
    """

    def __init__(self, resource, packet):
        # Initialize event
        super(ReceivePacket, self).__init__(resource._env)
        self.resource = resource
        self.packet = packet

        # Add this event to the packet queue
        resource._packets.append(self)
        resource._receive()

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        # If the event was interrupted, dequeue it
        if not self.triggered:
            self.resource._packets.remove(self)

    cancel = __exit__


class PacketQueue(object):
    """FIFO resource for processing packets.

    This class is a FIFO resource. It receives packets, but does not
    accept packet transmission requests.  Every packet arrival triggers
    a packet transmission, at which time the next packet in the queue is
    popped and returned.

    :param simpy.Environment env: the simulation environment
    :param int addr: the address of this resource
    """

    def __init__(self, env, addr):
        self._env = env
        # Router address
        self._addr = addr
        # Queue of packet events to process
        self._packets = list()

        # Bind event constructors as methods
        simpy.core.BoundClass.bind_early(self)

    receive = simpy.core.BoundClass(ReceivePacket)
    """Receive a packet."""

    @property
    def addr(self):
        """The address of this router."""
        return self._addr

    def _receive(self):
        """Receive a packet, and return a packet to be sent."""
        event = self._packets.pop(0)
        event.succeed(event.packet)


class HostResource(PacketQueue):
    """Resource representing a host.
    
    This is a FIFO resource which handles packets for a host.  If an
    inbound data packet is received, an acknowledgement is generated and
    sent, and the data packet itself is discarded (data packets don't
    have a destination flow)

    :param simpy.Environment env: the simulation environment
    :param int addr: the address of this host
    """

    def _receive(self):
        """Receive a packet, and return a packet to be sent."""
        event = self._packets.pop(0)

        # If a data packet has reached its destination
        if event.packet.dest == self.addr and event.packet.size == Packet.size:
            logger.info("host {} triggering acknowledgement for packet {} "
                        "at time {}".format(self._addr, event.packet.id, 
                                            self._env.now))
            
            # Send back an ackonwledgement packet
            event.succeed(event.packet.acknowledgement())
        else:
            # Transmit the packet
            event.succeed(event.packet)


class LinkTransport(simpy.events.Event):
    """Simulator event representing directional packet transmission.

    This event takes a resource as a parameter, and represents the
    transmission of a packet arcoss one direction of a full-duplex link,
    through the use of a limited size, drop-tail buffer.

    :param link: the link that this transport handler binds to
    :type link: :class:`LinkResource`
    :param bool direction: the (binary) direction of this transport handler
    :param packet: the packet to transport
    :type packet: :class:`Packet`
    """

    def __init__(self, link, direction, packet):
        # Initialize event
        super(LinkTransport, self).__init__(link._env)
        self._link = link
        self._direction = direction
        self._packet = packet

        # Enqueue the packet for transmission, if the buffer isn't full
        if link._fill[direction] + packet.size <= link.size:
            # Increment buffer fill
            link._fill[direction] += packet.size
            # Enqueue packet
            link._packet_queues[direction].put_nowait(self)
            # Transport a packet through the link
            link._receive(direction)

    def __enter__(self):
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
    allowed directions (0 or 1), each of which has a dedicated buffer.

    :param simpy.Environment env: the simulation environment
    :param int buf_size: link buffer size, in bits
    """

    def __init__(self, env, buf_size):
        self._env = env

        # Buffer size (bits)
        self._size = buf_size
        # Buffer fill (bits)
        self._fill = [0, 0]

        # Buffers for each edge direction
        self._packet_queues = (queue.Queue(), queue.Queue())
        # Bind any classes into methods now
        simpy.core.BoundClass.bind_early(self)

    transport = simpy.core.BoundClass(LinkTransport)
    """Transport packets across the link in a given direction."""

    @property
    def size(self):
        """Maximum buffer capacity in bits."""
        return self._size

    def fill(self, direction):
        """Returns the proportion of the buffer which is filled.

        :return: buffer fill as a proportion of buffer size
        :rtype: float
        """
        return self._fill[direction] / self._size

    def _receive(self, direction):
        """Dequeue a packet and send it."""
        # Dequeue a packet
        event = self._packet_queues[direction].get_nowait()
        # Update buffer fill
        self._fill[direction] -= event.packet.size
        # Return dequeued packet
        event.succeed(event.packet)

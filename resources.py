"""
.. module:: resources
    :platform: Unix
    :synopsis: This module defines network components as SimPy resources
"""

import heapq
import logging
import queue
import simpy
import simpy.util

from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOWN = 0
"""Download direction."""

UP = 1
"""Upload diretion."""


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
    """Packet size (bits)."""

    def __init__(self, src, dest, fid, pid, payload=None):
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
        """Return the packet's source address.

        :return: source address
        :rtype: int
        """
        return self._src

    @property
    def dest(self):
        """Return the packet's destination address.

        :return: destination address
        :rtype: int
        """
        return self._dest

    @property
    def flow(self):
        """Return the ID of the sender (flow) on the source host.

        :return: flow ID on localhost
        :rtype: int
        """
        return self._flow

    @property
    def id(self):
        """Return the ID of this packet.

        :return: packet ID for the source flow
        :rtype: int
        """
        return self._id

    @property
    def data(self):
        """Return this packet's payload of data.

        :return: the payload of this packet
        :rtype: object
        """
        return self._data

    def acknowledge(self, expected):
        """Generate an acknowledgement for this packet.

        :param int expected: the next packet id expected (ACK payload)
        :return: an acknowledgement packet matching this packet
        :rtype: :class:`ACK`
        """
        return ACK(self._dest, self._src, self._flow, self._id, expected)


class ACK(Packet):
    """This class represents an acknowledgement packet.

    :param src: source host
    :type src: :class:`process.Host`
    :param int dest: destination address
    :param int fid: source flow id
    :param int pid: packet id
    :param int expected: next expected packet id
    """

    # Simulated acknowledgement packet size (bits)
    size = 512
    """Packet size (bits)."""

    def __init__(self, src, dest, fid, pid, expected):
        # Packet source address
        self._src = src
        # Packet destination address
        self._dest = dest
        # Flow ID on the source host
        self._flow = fid
        # ID of the packet which triggered this acknowledgement
        self._id = pid
        # Acknowledgement packets have no payload
        self._data = expected


class RoutingPacket(Packet):
    """This class represents a routing packet.

    :param int pid: packet id
    :param object payload: packet payload
    """    

    # Simulated routing packet size (bits)
    size = 512
    """Packet size (bits)"""

    def __init__(self, pid, payload):
        # Packet source address
        self._src = None
        # Packet destination address
        self._dest = None
        # Flow ID on the source host
        self._flow = None
        # Packet ID
        self._id = pid
        # Packet data
        self._data = payload


class LinkTransport(simpy.events.Event):
    """SimPy event representing directional packet transmission.

    This event represents the transmission of a packet arcoss a
    full-duplex link in the given direction.  It uses a limited size,
    drop-tail buffer to queue packets for transmission.

    :param link: the link that this transport event binds to
    :type link: :class:`LinkResource`
    :param int direction: the direction of this transport event
    :param packet: the packet to transport
    :type packet: :class:`Packet`
    """

    def __init__(self, link, direction, packet):
        # Initialize event
        super(LinkTransport, self).__init__(link.env)
        self._link = link
        self._direction = direction
        self._packet = packet

        # Enqueue the packet for transmission, if the buffer isn't full
        if self._link._fill[direction] + self._packet.size <= self._link.size:
            # logger.info("enqueueing packet {}, {}, {} at time {}".format(
            #     self._packet.src, self._packet.flow, self._packet.id, 
            #     self._link.env.now))
            # Increment buffer fill
            self._link._fill[direction] += self._packet.size
            # Enqueue self._packet
            self._link._queues[direction].append(self)
        else:
            logger.info("dropped packet {}, {}, {} at time {}".format(
                self._packet.src, self._packet.flow, self._packet.id, 
                self._link.env.now))
        # Flush as much of the buffer as possible through the self._link
        self._link.flush(direction)

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        pass

    def cancel(self, exception, value, traceback):
        logger.info("{}\t{}".format(exception, value))
        self.__exit__()

    @property
    def direction(self):
        """Link direction for this transmission event.

        :return: link direction
        :rtype: int
        """
        return self._direction

    @property
    def packet(self):
        """The packet to be transmitted.

        :return: the packet in transit
        :rtype: :class:`Packet`
        """
        return self._packet


class LinkResource(object):
    """SimPy resource representing a link.

    This class is a full-duplex link implemented as a resource.  Packet
    transmission is parametrized by direction, and each link has two
    allowed directions (0 or 1), each of which has a dedicated buffer.
    New packets are added to the appropriate buffer, and trigger a buffer
    flush, which sends as much data from the buffer as possible through
    the link in the specified direction.

    :param simpy.Environment env: the simulation environment
    :param int size: link buffer size, in bits
    """

    def __init__(self, env, capacity, size):
        self.env = env
        # Link capacity (bps)
        self._capacity = capacity
        # Buffer size (bits)
        self._size = size
        # Buffer fill (bits)
        self._fill = [0, 0]
        # Link traffic (bps)
        self._traffic = [0, 0]

        # Buffers for each edge direction
        self._queues = (deque(), deque())
        # Bind any classes into methods now
        simpy.core.BoundClass.bind_early(self)

    transport = simpy.core.BoundClass(LinkTransport)
    """Transport packets across the link in a given direction."""

    @property
    def capacity(self):
        """The maximum bitrate of the link in bps."""
        return self._capacity

    @property
    def size(self):
        """Maximum buffer capacity in bits.

        :return: buffer size (bits)
        :rtype: int
        """
        return self._size

    def available(self, direction):
        """The available link capacity in the given direction.

        :param int direction: link direction
        :return: directional traffic (bps)
        :rtype: int
        """
        return self._capacity - self._traffic[direction]

    def update_traffic(self, direction, delta):
        """Update the link traffic in a given direction.

        Negative delta is equivalent to decreasing traffic.

        :param int direction: link direction
        :param int delta: change in directional traffic (bps)
        :return: None
        """
        # Check for valid value
        if delta < 0 and self._traffic[direction] < abs(delta):
            raise ValueError("not enough traffic")
        # Update traffic
        self._traffic[direction] += delta

    def fill(self, direction):
        """Returns the proportion of the buffer which is filled.

        :return: buffer fill as a proportion of buffer size
        :rtype: float
        """
        return self._fill[direction] / self._size

    def flush(self, direction):
        """Send as many packets as possible from the buffer.

        :param int direction: link direction to flush
        :return: the packets popped from the buffer
        :rtype: [:class:`Packet`]
        """
        # Initialize packet list
        flushed = list()
        event = None

        while len(self._queues[direction]) > 0:
            # Dequeue a packet
            event = self._queues[direction].popleft()
            # If the link isn't busy, flush another packet
            if event.packet.size + self._traffic[direction] <= self._capacity:
                flushed.append(event.packet)
                # Update buffer fill
                self._fill[direction] -= event.packet.size
            else:
                # Requeue the packet
                self._queues[direction].appendleft(event)
                break

        if flushed:
            event.succeed(flushed)


class ReceivePacket(simpy.events.Event):
    """SimPy event representing packet arrival.

    This event takes a resource as a parameter, and represents the 
    arrival of a packet at that resource.  It also triggers a packet
    transmission for each packet received.  This event may also be used
    as a context manager.

    :param resource: the packet's destination queue (resource)
    :type resource: :class:`PacketQueue`
    :param packet: a packet to receive
    :type packet: :class:`Packet`
    """

    def __init__(self, resource, packet):
        # Initialize event
        super(ReceivePacket, self).__init__(resource.env)
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
    """SimPy resource for queueing packets.

    This class is a FIFO SimPy resource. It receives packets, but does not
    accept packet transmission requests.  Every packet arrival triggers
    a packet transmission, at which time the next packet in the queue is
    popped and returned.

    :param simpy.Environment env: the simulation environment
    :param int addr: the address of this resource
    """

    def __init__(self, env, addr):
        self.env = env
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
        """The address of this resource.

        :return: resource address
        :rtype: int
        """
        return self._addr

    def _receive(self):
        """Receive a packet, yield, and return a packet to be sent."""
        event = self._packets.pop(0)
        event.succeed(event.packet)


class HostResource(PacketQueue):
    """SimPy resource representing a host.
    
    This is a FIFO resource which handles packets for a host.  If an
    inbound data packet is received, an acknowledgement is generated and
    sent, and the data packet itself is discarded (data packets don't
    have a destination flow).  Otherwise, packets emitted by local flows
    are transmitted into the network.

    :param simpy.Environment env: the simulation environment
    :param int addr: the address of this host
    """

    def __init__(self, env, addr):
        super(HostResource, self).__init__(env, addr)
        # A hash table mapping (src, flow) -> min heap of expected indices
        self._expected = dict()

    def _receive(self):
        """Receive a packet, yield, and return an outbound packet."""
        # Pop a packet from the queue
        event = self._packets.pop(0)
        # If a data packet has reached its destination
        if event.packet.dest == self.addr and event.packet.size == Packet.size:
            logger.info("host {} triggering acknowledgement for packet {}, {}"
                        " at time {}".format(self._addr, event.packet.flow,
                                             event.packet.id, self.env.now))
            # Get the event's ID in the hash table
            flow = (event.packet.src, event.packet.flow)
            # If the flow doesn't have an entry
            if flow not in self._expected.keys():
                # Initialize the min heap to expect the 0th packet first
                self._expected[flow] = [0]
            # Min heap for this flow
            heap = self._expected[flow]
            # If we got the packet we were expecting
            if event.packet.id == heap[0]:
                # And we haven't already received the next packet
                if heap[0] + 2 not in heap[1:3]:
                    # Set the heap to expect the next packet in the sequence
                    # (replaces heapq.heappoppush for this specific case)
                    heap[0] += 1
                else:
                    # Pop the ID of the received packet off the heap
                    heapq.heappop(heap)
            else:
                # Push the ID expected after this packet onto the heap
                heapq.heappush(event.packet.id + 1)
            # Pop any ID's for packet's we've already received
            while heap[0] + 1 in heap[1:3]:
                heapq.heappop(heap)
            # Return an acknowledgement with the next expected ID
            event.succeed(event.packet.acknowledge(heap[0]))
        else:
            # Return the packet popped from the queue
            event.succeed(event.packet)

    def _begin_bellman_ford(self, host_id):
        '''Called by the network process. Begins the process of updating '''

        #outline:
        #the network class calls this function it begins the routing update
        #process by creating a routing packet with the host_id and the cost
        #of the link connecting it to a host

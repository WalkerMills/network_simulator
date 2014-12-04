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

from collections import deque, OrderedDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DOWN = 0
"""Download direction."""

UP = 1
"""Upload diretion."""


class MonitoredEnvironment(simpy.Environment):
    """SimPy environment with monitoring.

    Processes may register identifiers, along with a getter (function)
    to be periodically called, and its return value recorded.  After the
    last event at a given time are processed, all monitored values are
    updated.  However, monitored values are only updated after an event
    occurs, i.e., if no events occur at a given time, no update is
    performed.

    :param int initial_time: simulation time to start at
    """

    def __init__(self, initial_time=0):
        super(MonitoredEnvironment, self).__init__(initial_time)
        # Dictionary mapping identifier -> [(time, monitored value)]
        self._monitored = dict()
        self._getters = dict()
        self._step = 10000000
        self._update_proc = self.process(self._update_registered())

    def _update_registered(self):
        while True:
            for name, (g, avg, nonzero) in self._getters.items():
                value = g() / self._step**avg
                if not nonzero or (nonzero and value != 0):
                    self._monitored[name].append((self.now, value))
            yield self.timeout(self._step)

    def monitored(self):
        """The timestamped values of all monitored attributes.

        :return: monitored attribute dict
        :rtype: {str: [(int, object)]}
        """
        return self._monitored

    def values(self, name):
        """The values for the given identifier.

        :param str name: the identifier to retrieve values for
        :return: timestamped values list
        :rtype: [(int, object)]
        """
        return self._monitored[name]

    def register(self, name, getter, avg=False, nonzero=False):
        self._getters[name] = (getter, avg, nonzero)
        self._monitored[name] = list()

    def update(self, name, value):
        try:
            self._monitored[name].append((self.now, value))
        except KeyError:
            self._monitored[name] = [(self.now, value)]


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

    # @id.setter
    def set_id(self, value):
        """Sets this packet's ID to a new value

        :param function value: new value of the ID
        :return: None
        """
        #logger.debug("setter called")
        self._id = value

    @property
    def data(self):
        """Return this packet's payload of data.

        :return: the payload of this packet
        :rtype: object
        """
        return self._data

    def acknowledge(self, expected):
        """Generate an acknowledgement for this packet.
        Packets return an id based on the next packet id expected 
        by the host.

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


class Routing(Packet):
    """This class represents a routing packet.

    :param object payload: packet payload
    """    

    # Simulated routing packet size (bits)
    size = 512
    """Packet size (bits)"""

    def __init__(self, payload):
        # Packet source address
        self._src = None
        # Packet destination address
        self._dest = None
        # Flow ID on the source host
        self._flow = None
        # Packet ID
        self._id = None
        # Packet data
        self._data = payload


class Finish(Routing):
    """This class represents a finish packet.

    Finish packets are used to communicate the termination condition
    for dynamic routing.

    :param object payload: packet payload
    """


class LinkEnqueue(simpy.events.Event):
    """SimPy event representing packet buffering.

    This event represents the enqueuing of a packet in one of a link's
    two buffers, as specified by ``direction``

    :param buffer_: the buffer that this enqueuing event binds to
    :type buffer_: :class:`LinkBuffer`
    :param int direction: the direction of the buffer
    :param packet: the packet to enqueue
    :type packet: :class:`Packet`
    """

    def __init__(self, buffer_, direction, packet):
        # Initialize event
        super(LinkEnqueue, self).__init__(buffer_.env)
        # Set queuing direction
        self._direction = direction

        # Enqueue the packet for transmission, if the buffer isn't full
        if packet.size <= buffer_._available(direction):
            # logger.debug("enqueuing packet {}, {}, {} at time {}".format(
            #     self._packet.src, self._packet.flow, self._packet.id, 
            #     self._buffer.env.now))
            # Increment buffer fill
            buffer_._update_fill(direction, packet.size)
            # Enqueue packet
            buffer_._queues[direction].append(packet)
            # Add this packet to the dict of recent packets
            buffer_._add_recent(direction, packet)
            # Set dropped flag to False
            dropped = False
            buffer_.env.update("Link fill,{}".format(buffer_.id), 
                               sum(len(q) for q in buffer_._queues))
        else:
            #logger.debug("dropped packet {}, {}, {} at time {}\t{}".format(
            #    packet.src, packet.flow, packet.id, buffer_.env.now,
            #    len(buffer_._queues[direction])))
            # Update dropped count
            buffer_._dropped += 1
            # Set dropped flag to True
            dropped = True
            buffer_.env.update("Dropped packets,{}".format(buffer_.id), 
                               buffer_.dropped)
        # Finish the event
        self.succeed(dropped)

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        pass

    cancel = __exit__

    @property
    def direction(self):
        """Link direction for this transmission event.

        :return: link direction
        :rtype: int
        """
        return self._direction


class LinkBuffer(object):
    """SimPy resource representing a link's buffers.

    This class is a full-duplex link implemented as a resource.  Packet
    transmission is parametrized by direction, and each link has two
    allowed directions (0 or 1), each of which has a dedicated buffer.
    New packets are added to the appropriate buffer, and trigger a buffer
    flush, which sends as much data from the buffer as possible through
    the link in the specified direction.

    :param simpy.Environment env: the simulation environment
    :param int size: link buffer size, in bits
    :param int lid: link id
    """

    # TODO: update docstring

    def __init__(self, env, size, lid):
        self.env = env
        # Buffer size (bits)
        self._size = float(size)
        # Link id
        self._id = lid

        # Buffers for each edge direction
        self._queues = (deque(), deque())
        # Length of time for calculating average arrival rate & queuing delay
        self._time = 500000000 # .5 sec
        # Recently received packets
        self._recent = (OrderedDict(), OrderedDict())
        # Buffer fill (bits)
        self._fill = [0.0, 0.0]
        # Number of dropped packets
        self._dropped = 0
        # Size of the last packet transmitted in each direction
        self.last_size = [Packet.size, Packet.size]

        # running total for buffer occupancy
        self._occupancy = 0
        self._avg_fill = 0
        self._last_update = 0

        # Bind event constructors as methods
        simpy.core.BoundClass.bind_early(self)

    enqueue = simpy.core.BoundClass(LinkEnqueue)
    """Enqueue a packet in the specified direction."""

    @property
    def id(self):
        """Link id."""
        return self._id

    @property
    def dropped(self):
        """The number of dropped packets

        :return: dropped packets
        :rtype: int
        """
        return self._dropped

    @property
    def size(self):
        """Maximum buffer capacity in bits.

        :return: buffer size (bits)
        :rtype: int
        """
        return self._size

    def _add_recent(self, direction, packet):
        """Add a packet to the dictionary of recent packets.

        If a packet is received which invalidates earlier data, remove
        that data.

        :param int direction: link direction
        :param packet: the packet to add
        :type packet: :class:`Packet`
        :return: None
        """
        if not isinstance(packet, Routing):
            self._recent[direction][packet] = [self.env.now, None]
        invalid = list()
        for p, times in self._recent[direction].items():
            if times[0] + self._time < self.env.now:
                invalid.append(p)
            else:
                break
        for p in invalid:
            del self._recent[direction][p]

    def _available(self, direction):
        """The available buffer capacity in the given direction.

        :param int direction: link direction
        :return: free buffer space (bits)
        :rtype: int
        """
        return self._size - self._fill[direction]

    def _enqueue(self, event):
        """Finish an equeuing."""
        event.succeed()

    def _update_fill(self, direction, delta):
        """Update buffer fill for the given direction.

        Negative delta is equivalent to decreasing buffer fill

        :param int direction: link direction
        :param int delta: change in buffer fill (bits)
        :return: None
        """
        # Check for valid value
        if delta < 0 and self._fill[direction] < abs(delta):
            raise ValueError("not enough data in buffer")
        # Update fill
        self._fill[direction] += delta

    def update_buffered(self, direction, time):
        diff = self._last_update - time
        if diff >= self._time:
            self._avg_fill = self._occupancy / diff
            self._occupancy = 0
        else:
            self._occupancy += len(self._queues[direction])
        #logger.debug("*****occupancy: {}, fill: {}".format(
        #            self._occupancy, self._avg_fill))


    def buffered(self, direction):
        """Total number of packets in link buffers."""
        return self._occupancy

    def dequeue(self, direction):
        """Dequeue a packet from the specified buffer.

        :param int direction: link direction
        :return: the dequeued packet
        :rtype: :class:`Packet` or None
        """
        try:
            # Get a packet from the queue
            packet = self._queues[direction].popleft()
            # Decrement buffer fill
            self._update_fill(direction, -packet.size)
            # Update last packet size
            self.last_size[direction] = packet.size
            if not isinstance(packet, Routing):
                # Set buffer exit time
                self._recent[direction][packet][1] = self.env.now
        except IndexError:
            # If there is no packet to dequeue, set packet to None
            packet = None
        return packet

    def fill(self, direction):
        """Returns the proportion of the buffer which is filled.

        :param int direction: link direction
        :return: buffer fill as a proportion of buffer size
        :rtype: float
        """
        return self._fill[direction] / self._size

    def queued(self, direction):
        """Estimated number of queued packets.

        This value is calculated using Little's law, which tells us that
        the expected number of queued packets is equal to the rate at
        which packets arrive, and the average amount of time each packet
        spends in the buffer.

        :param int direction: link direction
        :return: estimated number of queued packets
        :rtype: int
        """
        times = list(sorted(filter(
            lambda t: t[1], 
            self._recent[direction].values())))
        if len(times) == 1:
            return 1
        elif len(times) > 1:
            rate = len(times) / (times[-1][0] - times[0][0])
            delay = sum(t[1] - t[0] for t in times) / len(times)
            return rate * delay
        else:
            return 0


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
    """SimPy resource for queuing packets.

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
                heapq.heappush(heap, event.packet.id + 1)
            # Pop any ID's for packet's we've already received
            while heap[0] + 1 in heap[1:3]:
                heapq.heappop(heap)
            # Return an acknowledgement with the next expected ID
            event.succeed(event.packet.acknowledge(heap[0]))
        else:
            # Return the packet popped from the queue
            event.succeed(event.packet)

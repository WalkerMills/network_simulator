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

        # Dictionary mapping identifier -> getter
        self._getters = dict()
        # Dictionary mapping identifier -> [(time, monitored value)]
        self._monitored = dict()

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

    def _update(self):
        """Update all monitored values."""
        # For each identifier
        for name, getter in self._getters.items():
            # Append a new timestamped value to the list 
            self._monitored[name].append((self._now, getter()))

    def register(self, name, getter):
        """Register a new identifier.

        Raise a KeyError if the given identifier already exists.

        :param str name: the identifier
        :param function getter: a function to update the monitored value with
        :return: None
        """
        # Don't accept duplicate identifiers
        if name in self._getters.keys():
            raise KeyError("already monitoring {}".format(name))

        # Add this identifier to the getter & values dictionaries
        self._getters[name] = getter
        self._monitored[name] = []

    def step(self):
        """Process the next event, and update the monitored values.

        Raise an :exc:`simpy.core.EmptySchedule` if no further events
        are available.

        :return: None
        """
        try:
            self._now, _, _, event = heapq.heappop(self._queue)
        except IndexError:
            raise simpy.core.EmptySchedule()

        # Process callbacks of the event.
        for callback in event.callbacks:
            callback(event)
        event.callbacks = None

        # If the next event is in the future, or nonexistent 
        if self.peek() > self._now:
            # Update the monitored values
            self._update()

        if not event.ok and not hasattr(event, "defused"):
            # The event has failed, check if it is defused.
            # Raise the value if not.
            raise event._value


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
        logger.info("setter called")
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

    This event represents the enqueueing of a packet in one of a link's
    two buffers, as specified by ``direction``

    :param buffer_: the buffer that this enqueueing event binds to
    :type buffer_: :class:`LinkBuffer`
    :param int direction: the direction of the buffer
    :param packet: the packet to enqueue
    :type packet: :class:`Packet`
    """

    def __init__(self, buffer_, direction, packet):
        # Initialize event
        super(LinkEnqueue, self).__init__(buffer_.env)
        # Set queueing direction
        self._direction = direction

        # Enqueue the packet for transmission, if the buffer isn't full
        if packet.size <= buffer_._available(direction):
            # logger.info("enqueueing packet {}, {}, {} at time {}".format(
            #     self._packet.src, self._packet.flow, self._packet.id, 
            #     self._buffer.env.now))
            # Enqueue packet
            buffer_._queues[direction].append(packet)
            # Increment buffer fill
            buffer_._update_fill(direction, packet.size)
            # Set dropped flag to False
            dropped = False
        else:
            logger.info("dropped packet {}, {}, {} at time {}".format(
                packet.src, packet.flow, packet.id, buffer_.env.now))
            # Update dropped count
            buffer_._dropped += 1
            # Set dropped flag to True
            dropped = True
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
    """

    # TODO: update docstring

    def __init__(self, env, size):
        self.env = env
        # Buffer size (bits)
        self._size = float(size)

        # Buffers for each edge direction
        self._queues = (deque(), deque())
        # Buffer fill (bits)
        self._fill = [0.0, 0.0]
        # Number of dropped packets
        self._dropped = 0

        self._fill_cnt = list()

        # Bind event constructors as methods
        simpy.core.BoundClass.bind_early(self)

    enqueue = simpy.core.BoundClass(LinkEnqueue)
    """Enqueue a packet in the specified direction."""

    @property
    def buffered(self):
        """Return number of packets in link buffers"""
        return sum(len(q) for q in self._queues)

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

    def _available(self, direction):
        """The available buffer capacity in the given direction.

        :param int direction: link direction
        :return: free buffer space (bits)
        :rtype: int
        """
        return self._size - self._fill[direction]

    def _enqueue(self, event):
        """Finish an equeueing."""
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
        if not self._fill_cnt:
            self._fill_cnt = [[self.env.now, self.buffered]]
        elif self.env.now == self._fill_cnt[-1][0]:
            self._fill_cnt[-1][1] = max(self.buffered, self._fill_cnt[-1][1])
        else:
            self._fill_cnt.append([self.env.now, self.buffered])


    def dequeue(self, direction):
        """Dequeue a packet in from the specified buffer.

        :param int direction: link direction
        :return: the dequeued packet
        :rtype: :class:`Packet` or None
        """
        try:
            # Get a packet from the queue
            packet = self._queues[direction].popleft()
            # Decrement buffer fill
            self._update_fill(direction, -packet.size)
        except IndexError:
            # If there is no packet to dequeue, set packet to None
            packet = None
        return packet

    def fill(self, direction):
        """Returns the proportion of the buffer which is filled.

        :return: buffer fill as a proportion of buffer size
        :rtype: float
        """
        return self._fill[direction] / self._size


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
                heapq.heappush(heap, event.packet.id + 1)
            # Pop any ID's for packet's we've already received
            while heap[0] + 1 in heap[1:3]:
                heapq.heappop(heap)
            # Return an acknowledgement with the next expected ID
            event.succeed(event.packet.acknowledge(heap[0]))
        else:
            # Return the packet popped from the queue
            event.succeed(event.packet)

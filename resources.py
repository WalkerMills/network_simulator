"""
.. module:: resources
    :platform: Unix
    :synopsis: This module defines network components as SimPy resources
"""

import heapq
import queue
import simpy
import simpy.util

from collections import deque

DOWN = 0
"""Download direction."""

UP = 1
"""Upload diretion."""

def reset(klass):
    """Class decorator for adding a reset method.

    The reset method takes an object, the name of an attribute, and
    optionally an index, retrieves the value of the attribute (at the
    index, if given), resets the value using its type constructor, and
    returns the original value.  This class decorator is primarily for
    use with :func:`MonitoredEnvironment.register`; decorated classes 
    may use the reset method to generically create a getter for a time-
    averaged value, since it both fetches and resets the value, as long
    as that value is stored in a class attribute.

    :param class klass: class to decorate
    :return: the decorated class
    :rtype: class
    """

    def _reset(self, attr, index=None):
        """Get the value of the specified attribute, and reset it.

        :param str attr: attribute name
        :param object index: index in the container referenced by ``self.attr``
        :return: ``self.attr``, before resetting its value
        :rtype: object
        """
        if index is not None:
            # Get the value of the attribute
            val = getattr(self, attr)
            # Extract the return value
            ret = val[index]
            # Reset the replacement value at the index
            val[index] = type(ret)()
        else:
            # Get the return value
            ret = getattr(self, attr)
            # Reset the replacement value
            val = type(ret)()
        # Replace the value of the attribute
        setattr(self, attr, val)
        # Return the original value
        return ret

    # Add the reset method to the new type's __dict__ attribute
    d = dict(klass.__dict__)
    d["reset"] = _reset
    # Return the new type, with the same name, bases, and updated __dict__
    return type(klass.__name__, klass.__bases__, d)


class MonitoredEnvironment(simpy.core.Environment):
    """SimPy environment with monitoring.

    Processes may register identifiers, along with a getter (function)
    to be periodically called, and its return value recorded.
    Alternatively, processes may use the ``update`` method to add a value
    for the given identifier.  Thus, the environment supports both periodic
    and process-controlled monitoring.

    :param int initial_time: simulation time to start at
    :param int step: registered value update period (ns)
    """

    def __init__(self, initial_time=0, step=100000000):
        super(MonitoredEnvironment, self).__init__(initial_time)
        # Dictionary mapping identifier -> [(time, monitored value)]
        self._monitored = dict()
        self._getters = dict()
        self._step = step

    @property
    def monitored(self):
        """The timestamped values of all monitored attributes.

        :return: monitored attribute dict
        :rtype: {str: [(int, object)]}
        """
        return self._monitored

    def _update_process(self, name, getter, avg, step):
        """Process for periodically updating an identifier."""
        while True:
            # Call the getter, and average its value if the avg flag is set
            value = getter() / step**avg
            # Record a timestamped value
            self.update(name, value)
            # Wait for the next update
            yield self.timeout(step)

    def register(self, name, getter, avg=False, step=None):
        """Register a new identifier.

        If the avg flag is set, the return value of the getter is divided
        by the time step over which the update is performed, in order to
        yield an average.  The environment is initialized with a default
        step, but the ``step`` parameter may override it for the given
        identifier.

        :param str name: identifier
        :param function getter: getter function
        :param bool avg: flag indicating whether to average the getter value
        :param int step: update period for this identifier
        :return: None
        """
        if name in self._monitored.keys():
            raise ValueError("duplicate identifier \"{}\"".format(name))
        # If no step was given
        if step is None:
            # Use the default step
            step = self._step
        # Start the update process for this value
        self._getters[name] = self.process(
            self._update_process(name, getter, avg, step))

    def update(self, name, value):
        """Add a value for the specified identifier.

        If the given identifier does not exist, an entry for it is created.

        :param str name: identifier
        :param object value: the value to add
        :return: None
        """
        try:
            # Record the timestamped value for this identifier
            self._monitored[name].append((self.now, value))
        except KeyError:
            # Create a new entry if none exists
            self._monitored[name] = [(self.now, value)]

    def values(self, name):
        """The values for the given identifier.

        :param str name: the identifier to retrieve values for
        :return: timestamped values list
        :rtype: [(int, object)]
        """
        return self._monitored[name]


class Packet:
    """This class represents a packet in our simulation.

    :param int src: source host id
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
        """The packet's source address.

        :return: source address
        :rtype: int
        """
        return self._src

    @property
    def dest(self):
        """The packet's destination address.

        :return: destination address
        :rtype: int
        """
        return self._dest

    @property
    def flow(self):
        """The ID of the sender (flow) on the source host.

        :return: flow ID on localhost
        :rtype: int
        """
        return self._flow

    @property
    def id(self):
        """The ID of this packet.

        :return: packet ID for the source flow
        :rtype: int
        """
        return self._id

    @property
    def data(self):
        """The packet's payload of data.

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

    :param int src: source host id
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

    Finish packets communicate that a router has locally converged, and
    are used to signal the termination condition for Bellman-Ford
    dynamic routing.

    :param object payload: packet payload
    """


class LinkEnqueue(simpy.events.Event):
    """SimPy event representing packet buffering.

    This event represents the enqueuing of a packet in one of a link's
    two buffers, as specified by ``direction``.  This event may also be
    used as a context manager.

    :param buffer_: the buffer that this enqueuing event binds to
    :type buffer_: :class:`LinkBuffer`
    :param int direction: the direction of the buffer
    :param packet: the packet to enqueue
    :type packet: :class:`Packet`
    """

    def __init__(self, buffer_, direction, packet):
        # Initialize event
        super(LinkEnqueue, self).__init__(buffer_.env)
        # Enqueue the packet for transmission, if the buffer isn't full
        if packet.size <= buffer_._available(direction):
            # Increment buffer fill
            buffer_._update_fill(direction, packet.size)
            # Enqueue packet
            buffer_._queues[direction].append(packet)
            # Set dropped flag to False
            dropped = False
        else:
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


class LinkBuffer:
    """SimPy resource representing a link's buffers.

    This class is a full-duplex link implemented as a resource.  Packet
    transmission is parametrized by direction, and every link has two
    allowed directions (:data:`resources.DOWN` or :data:`resources.UP`),
    each of which has a dedicated buffer.  New packets are added to the
    appropriate buffer, to be dequeued by the :class:`process.Link` that
    owns the resource instance.

    :param simpy.Environment env: the simulation environment
    :param int size: link buffer size, in bits
    :param int lid: link id
    """

    def __init__(self, env, size, lid):
        self.env = env
        # Buffer size (bits)
        self._size = float(size)
        # Link id
        self._id = lid

        # Buffers for each edge direction
        self._queues = (deque(), deque())
        # Buffer fill (bits)
        self._fill = [0.0, 0.0]
        # Number of dropped packets
        self._dropped = 0
        # Running total of buffer occupancy
        self._occupancy = [0, 0]
        # Monitor buffer fill periodically
        self.env.register("Link fill,{},{}".format(self.id, DOWN), 
                          lambda: len(self._queues[DOWN]))
        self.env.register("Link fill,{},{}".format(self.id, UP), 
                          lambda: len(self._queues[UP]))
        # Bind event constructors as methods
        simpy.core.BoundClass.bind_early(self)

    enqueue = simpy.core.BoundClass(LinkEnqueue)
    """Enqueue a packet in the specified direction.

    This is a SimPy event, which is bound as a method to each ``LinkBuffer``
    object upon its instantiation.  When this event is yielded, it also
    returns a flag whose truth value indicates whether the packet was
    dropped (buffer was full) or not.

    :param int direction: link direction
    :param packet: packet to enqueue
    :type packet: :class:`Packet`
    :return: dropped flag
    :rtype: bool
    """

    @property
    def id(self):
        """Link id."""
        return self._id

    @property
    def dropped(self):
        """The number of dropped packets.

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
        return self.size - self._fill[direction]

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

    def buffered(self, direction):
        """Cumulative number of packets in a link buffer.

        :param int direction: link direction
        :return: cumulative buffer occupancy
        :rtype: int
        """
        return self._occupancy[direction]

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
        except IndexError:
            # If there is no packet to dequeue, set packet to None
            packet = None
        return packet

    def update_buffered(self, direction, time):
        """Update the cumulative buffer occupancy.

        :param int direction: link direction
        :param int time: time of the update
        :return: None
        """
        self._occupancy[direction] += len(self._queues[direction])


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


class PacketQueue:
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
    """Receive a packet.

    This is a SimPy event, which is bound as a method to each ``PacketQueue``
    object upon its instantiation.  When this event is yielded, it also
    returns a packet popped from the queue.

    :param int direction: link direction
    :param packet: packet to enqueue
    :type packet: :class:`Packet`
    :return: a dequeued packet
    :rtype: :class:`Packet`
    """

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
            # Pop any ID"s for packet"s we've already received
            while heap[0] + 1 in heap[1:3]:
                heapq.heappop(heap)
            # Return an acknowledgement with the next expected ID
            event.succeed(event.packet.acknowledge(heap[0]))
        else:
            # Return the packet popped from the queue
            event.succeed(event.packet)

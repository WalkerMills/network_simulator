"""
.. module:: process
    :platform: Unix
    :synopsis: This module defines network actors as processes
"""

import logging
import math
import queue
import random
import simpy

import resources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Flow(object):
    """SimPy process representing a flow.

    Each flow process is connected to a source host, and generates as
    many packets as are necessary to send all of its data.  If data is
    None, a random, 8-bit number of packets are sent.

    :param simpy.Environment env: the simulation environment
    :param host: the source host of this flow
    :type host: :class:`Host`
    :param int dest: the address of the destination host
    :param int data: the total amount of data to transmit (bits)
    :param int window: the transmission window size (bits)
    :param int timeout: the time to wait for :class:`resources.ACK`'s 
    """

    def __init__(self, env, host, dest, data, window, timeout, delay):
        self._env = env
        # Flow host
        self._host = host
        # FLow destination
        self._dest = dest
        # Amount of data to transmit (bits)
        self._data = data
        # Window size for this flow
        self._window = window
        # Time (simulation time) to wait for an acknowledgement before
        # retransmitting a packet
        self._time = timeout
        # Time (simulation time) to wait before initial transmission
        self._delay = delay

        # Register this flow with its host, and get its ID
        self._id = self._host.register(self)
        # Initialize packet generator
        self._packets = self._generator()
        # Dictionary of unacknowedged packets (maps ID -> packet)
        self._outbound = dict()
        # Number of packets generated so far
        self._sent = 0
        # Initialize per-window sent data counter
        self._window_sent = 0
        # Wait to receive acknowledgement (passivation event)
        self._wait_for_ack = self._env.event()

    @property
    def dest(self):
        """The destination address for this flow.

        :return: destination address
        :rtype: int
        """
        return self._dest

    @property
    def data(self):
        """The total amount of data to transfer.

        :return: data size (bits)
        :rtype: int
        """
        return self._data

    @property
    def id(self):
        """The ID of this flow, assigned by its host.

        :return: flow id, relative to localhost
        :rtype: int
        """
        return self._id

    @property
    def timeout(self):
        """The time after which packets are considered dropped.

        :return: acknowledgement timeout (simulation time)
        :rtype: int
        """
        return self._time

    @property
    def window(self):
        """Transmission window size, in bits.

        :return: window size (bits)
        :rtype: int
        """
        return self._window

    def _generator(self):
        """Create a packet generator.

        If a data size is given, return a generator which yields
        ceil(data size / packet size) packets.  If data size is None,
        it yields a random, 8-bit number of packets.
        """
        n = 0
        if self._data == None:
            # Pick a random 8-bit number of packets to send
            n = random.getrandbits(8)
        else:
            # Calculate how many packets are needed to send self._data bits
            n = math.ceil(self._data / resources.Packet.size)

        # Create packet generator
        g = (resources.Packet(self._host.addr, self._dest, self._id, i)
             for i in range(n))

        return g

    def generate(self):
        """Generate and transmit outbound data packets.

        This process has an underlying generator which yields the number
        of packets required to transmit all the data.  The process then
        passivates until the acknowledgement of those packets reactivates
        it.

        :return: None
        """
        # Yield initial delay
        yield self._env.timeout(self._delay)

        # Send all the packets
        while True:
            try:
                # Send as many packets as fit in our window
                while self._window_sent < self._window:
                    logger.info(
                        "flow {}, {} sending packet {} at time {}".format(
                            self._id, self._host.addr, self._sent, 
                            self._env.now))
                    # Generate the next packet to send
                    packet = next(self._packets)
                    # Increment the count of generated packets
                    self._sent += 1
                    # Increment the counter for data sent in this window
                    self._window_sent += packet.size
                    # Add the packet to the map of unacknowedged packets
                    self._outbound[packet.id] = packet
                    # Send the packet through the connected host
                    yield self._env.process(self._host.receive(packet))
                # Passivate packet generation
                yield self._wait_for_ack
                # On reactivation, reset data counter for new window
                self._window_sent = 0
            except StopIteration:
                # Stop trying to send packets once we run out
                break
        # Wait for the packets in the last window
        yield self._wait_for_ack
        logger.info("flow {}, {} finished transmitting".format(self._host.addr,
                                                               self._id))

    def acknowledge(self, ack):
        """Receive an acknowledgement packet.

        If an acknowledgement is received from a packet in the previous
        window, this generator waits <timeout> units of simulation time
        until declaring any unacknowedged packets dropped, and
        retransmitting them.  Once finished transmitting any dropped
        packets, packet generation is reactivated.  If an acknowledgement
        is received from a retransmitted packet (i.e., outside the
        previous window), it is marked as acknowledged, but does not
        trigger a timeout or packet generation.

        :param ack: acknowledgement packet
        :type ack: :class:`resources.ACK`
        :return: None
        """
        logger.info("flow {}, {} acknowledges packet {} at time {}".format(
            self._id, self._host.addr, ack.id, self._env.now))

        # Acknowledge the packet whose ACK was received
        del self._outbound[ack.id]

        # If this ACK is not from a re-transmitted packet
        if ack.id >= self._sent - self._window:
            # Wait for other acknowledgements
            yield self._env.timeout(self._time)

            # TODO: congestion control algorithm should modify window
            #       and detect/recover dropped packets

            # Continue sending more packets
            self._wait_for_ack.succeed()
            self._wait_for_ack = self._env.event()

    def _tcp_reno(self):
        self._window += 8192
        logger.info("flow {}, {} window size increases to {} at time"
                    " {}".format(self._host.addr,self._id, self._window, 
                                 self._env.now))        


class Host(object):
    """SimPy process representing a host.

    Each host process has an underlying HostResource which handles
    (de)queuing packets.  This process handles interactions between the
    host resource, any active flows, and up to one outbound link.

    :param simpy.Environment env: the simulation environment
    :param int addr: the address of this host
    """

    def __init__(self, env, addr):
        # Initialize host resource
        self.res = resources.HostResource(env, addr)
        # Host address
        self._addr = addr
        # Active flows
        self._flows = list()
        # Outbound link
        self._transport = None

    @property
    def addr(self):
        """The address of this host.

        :return: host address
        :rtype: int
        """
        return self._addr

    @property
    def flows(self):
        """A list of flows active on this host.

        :return: a list of flows
        :rtype: [:class:`Flow`]
        """
        return self._flows

    def connect(self, transport):
        """Connect a new (link) transport handler to this host.

        :param transport: transport handler
        :type transport: :class:`Transport`
        :return: None
        """
        self._transport = transport

    def disconnect(self, transport):
        """Disconnect an existing transport handler.

        If the given transport handler is not connected to this host,
        do nothing.

        :param transport: the transport handler to disconnect
        :type transport: :class:`Transport`
        :return: None
        """
        if transport == self._transport:
            self._transport = None

    def receive(self, packet):
        """Receive a packet.

        Packets may be inbound or outbound, data or acknowledgement
        packets.  Inbound data packets automatically generate an
        outbound acknowledgement.

        :param packet: the packet to process
        :type packet: :class:`resources.Packet`
        """
        logger.info("host {} received packet {}, {} at time {}".format(
            self.addr, packet.flow, packet.id, self.res._env.now))
        # Queue new packet for transmission, and dequeue a packet. The
        # HostResource.receive event returns an outbound ACK if the
        # dequeued packet was an inbound data packet
        packet = yield self.res.receive(packet)

        # Transmit the dequeued packet
        yield self.res._env.process(self.transmit(packet))

    def transmit(self, packet):
        """Transmit a packet.

        Inbound data packets are replaced by an acknowledgement packet
        when they exit the host's internal queue, so there are only two
        cases: packets may be outbound, or they are ACK's destined for
        a flow on this host.

        :param packet: the outbound packet
        :type packet: :class:`resources.Packet`
        :return: None
        """

        if packet.dest != self._addr:
            logger.info("host {} transmitting packet {}, {}, {} at time"
                        " {}".format(self.addr, packet.src, packet.flow, 
                                     packet.id, self.res._env.now))
            # Transmit an outbound packet
            yield self.res._env.process(self._transport.send(packet))
        else:
            logger.info("host {} transmitting ACK {}, {}, {} at time"
                        " {}".format(self.addr, packet.src, packet.flow,
                                     packet.id, self.res._env.now))
            # Send an inbound ACK to its destination flow
            yield self.res._env.process(
                self._flows[packet.flow].acknowledge(packet))

    def register(self, flow):
        """Register a new flow on this host, and return the flow ID.

        :param flow: a new flow to register with this host
        :type flow: :class:`Flow`
        :return: None
        """
        self._flows.append(flow)
        return len(self._flows) - 1


class Transport(object):
    """This class is a directional transport handler for a link.

    Direction should be one of :data:`resources.UP` or :data:`resources.DOWN`

    :param link: the underlying link
    :type link: :class:`Link`
    :param int direction: packet transmission direction
    """

    def __init__(self, link, direction):
        self._link = link
        self._direction = direction

    def __eq__(self, other):
        return self._link == other.link and self._direction == other.direction

    def __hash__(self):
        return self._link.__hash__() * (-1)**(self._direction)

    @property
    def link(self):
        """The link that owns this transport handler."""
        return self._link

    @property
    def direction(self):
        """The direction of this transport handler."""
        return self._direction

    @property
    def cost(self):
        """Directional cost of the underlying link."""
        return self._link.cost(direction)

    def send(self, packet):
        """Send a packet across the link.

        :param packet: the packet to send
        :type packet: :class:`resources.Packet`
        :return: :method:`Link.receive` generating method
        :rtype: generator
        """
        return self._link.receive(self._direction, packet)


class Link(object):
    """SimPy process representing a link.

    Each link process has an underlying LinkResource which handles
    (de)queuing packets.  This process handles interactions between the
    link resource, and the processes it may connect.

    :param simpy.Environment env: the simulation environment
    :param int capacity: the link rate, in bits per second
    :param int size: the link buffer size, in bits
    :param int delay: the link delay in simulation time
    """

    def __init__(self, env, capacity, size, delay):
        # Initialize link resource
        self.res = resources.LinkResource(env, capacity, size)
        # Link delay (simulation time)
        self._delay = delay

        # Endpoints for each direction
        self._endpoints = [None, None]
        # "Upload" handler
        self._up = Transport(self, resources.UP)
        # "Download" handler
        self._down = Transport(self, resources.DOWN)

    @property
    def delay(self):
        """The link delay in seconds."""
        return self._delay

    @property
    def endpoints(self):
        """A list of connected endpoints (up to 1 per direction)"""
        return self._endpoints

    def cost(self, direction):
        """Return the cost of a direction on this link.

        Total cost is calculated as the product of propagation delay,
        (data) packet size, and 1 + buffer fill proportion, divided by
        the logarithm of available capacity.

        :param int direction: link direction to compute cost for

        """
        try:
            return self._delay * resources.Packet.size * \
                (1 + self.res.fill(direction)) / self.res.available(direction)
        except ValueError:
            # Available capacity was 0, so return infinite cost
            return float("inf")

    def connect(self, A, B):
        """Connect two network components via this link.

        :param A: the first new endpoint of this link
        :type A: :class:`Host`, or :class:`Router` 
        :param B: the second new endpoint of this link
        :type B: :class:`Host`, or :class:`Router` 
        :return: None
        """
        # Store the endpoints
        self._endpoints = [A, B]
        # Connect the "upload" (0 -> 1) hander
        self._endpoints[0].connect(self._up)
        # Connect the "download" (1 -> 0) handler 
        self._endpoints[1].connect(self._down)

    def disconnect(self):
        """Disconnect a link from its two endpoints.

        :return: None
        """
        # Disconnect "upload" handler
        self._endpoints[0].disconnect(self._up)
        # Disconnect "download" handler
        self._endpoints[1].disconnect(self._down)
        # Reset the endpoints
        self._endpoints = [None, None]

    def receive(self, direction, packet):
        """Receive a packet to transmit in a given direction.

        New packets are appended to one of the link's internal directional
        buffer, and trigger the dequeuing of a packet from that buffer.
        The dequeued packet is then sent across the link.  All link
        buffers are drop-tail.

        :param int direction: the direction to transport the packet
        :param packet: the packet to send through the link
        :type packet: :class:`resources.Packet`
        :return: None
        """
        logger.info("link received packet {}, {}, {} at time {}".format(
            packet.src, packet.flow, packet.id, self.res._env.now))
        # Queue the new packet for transmission, and dequeue a packet
        packets = yield self.res.transport(direction, packet)
        # Transmit the dequeued packets, if any
        yield self.res._env.process(self.transmit(direction, packets))

    def transmit(self, direction, packets):
        """Transmit packets across the link in a given direction.

        This generating function yields a transmission process for each
        packet given

        :param int direction: the direction to transport the packet
        :param packet: a list of packets to transmit
        :type packet: [:class:`resources.Packet`]
        :return: None
        """
        # Transmit packet after waiting, as if sending the packet
        # across a physical link
        for p in packets:
            logger.info("transmitting packet {}, {}, {} at time {}".format(
                p.src, p.flow, p.id, self.res._env.now))
            # Increment directional traffic
            self.res.update_traffic(direction, p.size)
            yield simpy.util.start_delayed(self.res._env,
                self._endpoints[direction].receive(p), self._delay)
            self.res.update_traffic(direction, -p.size)


class Router(object):
    """Simpy process representing a router.

    Router objects each have an underlying PacketQueue resource which
    receives all inbound packets.  When a new packet is received, it
    triggers a packet transmission event.  The dequeued packet is routed
    based on its destination, and data packets are sent to the appropriate
    transport handler (link).

    :param simpy.Environment env: the simulation environment
    :param int addr: the address of this router
    """

    def __init__(self, env, addr):
        # Initialize underlying PacketQueue
        self.res = resources.PacketQueue(env, addr)
        # Address of this router (router ID's are independent of host ID's)
        self._addr = addr
        # Set of outbound links
        self._links = set()
        # Dictionary mapping outbound links to destinations
        self._table = dict()

    @property
    def addr(self):
        """The address of this router.

        :return: router address
        :rtype: int
        """
        return self._addr

    def connect(self, transport):
        """Connect an outbound link.

        :param transport: transport handler
        :type transport: :class:`Transport`
        :return: None
        """
        # Connect a new transport handler to a new "port"
        self._links.add(transport)

    def disconnect(self, transport):
        """Disconnect a link, if it exists.

        :param transport: the transport handler to disconnect
        :type transport: :class:`Transport`
        :return: None
        """
        # If the given transport handler is connected, remove it
        self._links.discard(transport)
        try:
            # Delete this outbound link from the routing table
            del self._table[transport]
        except KeyError:
            # If it doesn't exist, do nothing
            pass

    def receive(self, packet):
        """Receive a packet, and yield a transmission event.

        :param packet:
        :type packet: :class:`resources.Packet`
        :return: None
        """
        logger.info("router {} received packet {} at time {}".format(
            self.addr, packet.id, self.res._env.now))
        # Push another packet through the queue
        packet = yield self.res.receive(packet)
        # Transmit the dequeued packet
        yield self.res._env.process(self.transmit(packet))

    def _route(self, address):
        """Return the correct transport handler for the given address.

        :param int address: the destination address
        :return: the transport handler selected by the routing policy
        :rtype: function
        """
        
        # TODO: return a transport handler based on the given address

        pass

    def transmit(self, packet):     
        """Transmit an outbound packet.        
       
        :param packet: the outbound packet     
        :type packet: :class:`resources.Packet`        
        :return: None      
        """        
        # Determine which transport handler to pass this packet to     
        transport = self._route(packet.addr)       
        # Send the packet      
        yield self.res._env.process(transport.send(packet))

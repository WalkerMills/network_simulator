"""
.. module:: process
    :platform: Unix
    :synopsis: This module defines network actors as processes
"""

import itertools
import logging
import math
import queue
import random
import simpy

from collections import deque

import resources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAST(object):
    """SimPy process implementing FAST TCP.

    :param flow: the flow to link this TCP instance to
    :type flow: :class:`Flow`
    :param int window: initial window size (in packets)
    :param int timeout: packet acknowledgement timeout (simulation time)
    :param int alpha: desired number of enqueued packets at equilibrium
    :param float gamma: window update weight
    """

    def __init__(self, flow, window, timeout, alpha, gamma=0.5):
        # The flow running this TCP algorithm
        self._flow = flow
        # Window size (packets)
        self._window = window
        # Time after which packets are considered dropped
        self._timeout = timeout
        # Packets in buffer at equilibrium
        self._alpha = alpha
        # Window update weight
        self._gamma = gamma

        # Mean round trip time
        self._mean_trip = float("inf")
        # Minimum observed round trip time
        self._min_trip = float("inf")
        # Estimated queueing delay
        self._delay = float("inf")

        # Packet generator
        self._gen = self._flow.generator()
        # Hash table mapping unacknowledged packet -> departure time
        self._unacknowledged = dict()
        # Dropped packet (retransmission) queue
        self._dropped = queue.Queue()

        # Next ID's expected by the destination host.  Last 3 are cached to
        # check for packet dropping
        self._next = deque(maxlen=3)

        self._window_ctrl_proc = None

    @property
    def window(self):
        """Transmission window size, in bits.

        :return: window size (bits)
        :rtype: int
        """
        return self._window

    @property
    def timeout(self):
        """The time after which packets are considered dropped.

        :return: acknowledgement timeout (simulation time)
        :rtype: int
        """
        return self._time

    def _estimate(self, trip):
        # Calculate weight for round trip mean update
        eta = min(3.0 / self._window, 0.25)
        # If the mean is unset
        if self._mean_trip == float("inf"):
            # Set the mean trip to the given value
            self._mean_trip = trip
        else:
            # Update moving average of round trip mean
            self._mean_trip = (1 - eta) * self._mean_trip + eta * trip

        # Update minimum round trip time, if necessary
        if trip < self._min_trip:
            self._min_trip = trip

        # Update estimated queueing delay
        self._delay = self._mean_trip - self._min_trip

    def _update_dropped(self):
        # For each unacknowledged packet
        for packet, time in list(self._unacknowledged.items())[:]:
            # If it has timed out
            if time <= self._flow.env.now - self._timeout:
                # Mark it as dropped
                self._dropped.put(packet)
                # Remove it from the unacknowledged packets
                del self._unacknowledged[packet]

    def _update_window(self):
        # Calculate the new window size, as directed by FAST TCP spec
        window = (1 - self._gamma) * self._window + self._gamma * \
            (self._window * self._min_trip / self._mean_trip + self._alpha)
        # Update window size by at most doubling it
        self._window = min(2 * self._window, window)

    def _window_size(self):
        # If there are no unacknowledged packets
        if len(self._unacknowledged) == 0:
            # The window is empty
            return 0
        # Sort the unacknowledged ID's
        ids = sorted(packet.id for packet in self._unacknowledged.keys())
        # Current window size = difference between the largest & smallest
        # ID's + 1
        return ids[-1] - ids[0] + 1

    def acknowledge(self, ack):
        """Process an acknowledgement packet.

        :param ack: the acknowledgement packet
        :type ack: :class:`resources.ACK`
        :return: None
        """
        # ID of the packet being acknowledged
        pid = ack.id
        # ID expected next by the destination host
        expected = ack.data

        # Push this into the deque (if the deque is full, this is
        # equivalent to self._next.popleft() before appending)
        self._next.append(expected)
        # Get the unacknowledged packet
        packet = list(filter(lambda p: p.id == pid, 
                      self._unacknowledged.keys()))[0]
        # Calculate round trip time
        rtt = self._flow.env.now - self._unacknowledged[packet]
        # Mark packet as acknowledged
        del self._unacknowledged[packet]
        # Update mean round trip time & queueing delay
        self._estimate(rtt)

        # If we have 3 dropped ACK's
        if self._next.count(self._next[0]) == 3:
            # Enter recovery
            yield self._flow.env.process(self.recover())

        if self._gen is not None:
            # Try to inject packets into the network
            yield self._flow.env.process(self.burst())

    def burst(self, recover=False):
        """Inject packets into the network.

        If recover is True, then the packets are taken from the dropped
        packet queue.

        :param bool recover: flag for recovery mode
        :return: None
        """
        # If we are in recovery mode
        if recover:
            # Take packets from the dropped packet queue, as long as there
            # are packets to take.  If there are multiple simultaneous
            # burst processes, each will take as many packets as they can
            # consume before the queue is exhausted
            gen = itertools.takewhile(
                lambda _: self._dropped.qsize() >= 0,
                (self._dropped.get_nowait() for i in range(
                    self._dropped.qsize())))
        else:
            # Otherwise, take packets from the data packet generator
            gen = self._gen
        try:
            # While the window is not full
            while self._window_size() < self._window:
                # Get the next data packet
                packet = next(gen)
                # Mark the packet as unacknowledged
                self._unacknowledged[packet] = self._flow.env.now
                # Transmit the packet
                yield self._flow.env.process(self._flow.transmit(packet))
                # If we are in recovery mode
                if recover:
                    # Update the window (size)
                    self._update_window()
        # Once we have exhausted the packet generator
        except StopIteration:
            # If we exhausted the data packet generator
            if not recover:
                # Set the generator to None
                self._gen = None
                # While there remains dropped packets
                while not self._dropped.empty():
                    # Retransmit as many dropped packets as possible
                    self.recover()
                    # Wait for acknowledgements
                    yield self._flow.env.timeout(self._timeout)
                    # Update dropped packets
                    self._update_dropped()
                try:
                    # Kill the window control process
                    self._window_ctrl_proc.interrupt()
                except RuntimeError:
                    pass

    def recover(self):
        """Enter recovery mode.

        :return: None
        """
        # Halve the window size
        self._window /= 2
        # Retransmit as many dropped packets as possible
        yield self._flow.env.process(self.burst(recover=True))

    def window_control(self):
        """Periodically update the window size.

        :return: None
        """
        try:
            while True:
                # Wait for acknowledgements
                yield self._flow.env.timeout(self._timeout)
                # Update dropped packets
                self._update_dropped()
                # If there exist dropped packets
                if not self._dropped.empty():
                    # Enter recovery mode (updates window size)
                    yield self._flow.env.process(self.recover())
                else:
                    # Otherwise, just update the window size
                    self._update_window()
        except simpy.events.Interrupt:
            pass

    def run(self):
        """Run the TCP algorithm (send all data packets).

        :return: None
        """
        # Start the window control process
        self._window_ctrl_proc = self._flow.env.process(self.window_control())
        # Yield a burst process
        yield self._flow.env.process(self.burst())


class Flow(object):
    """SimPy process representing a flow.

    Each flow process is connected to a source host, and generates as
    many packets as are necessary to send all of its data.  If data is
    None, a random, 8-bit number of packets are sent.  Every flow needs
    a TCP algorithm to be specified, from among the currently supported
    TCP algorithms.  At the moment, the only allowed specifier is 
    \"FAST\".  See :class:`FAST` for details on what tcp_params should
    look like.

    :param simpy.Environment env: the simulation environment
    :param host: the source host of this flow
    :type host: :class:`Host`
    :param int dest: the address of the destination host
    :param int data: the total amount of data to transmit (bits)
    :param int delay: the simulation time to wait before sending any packets
    :param str tcp: TCP algorithm specifier
    :param list tcp_params: parameters for the TCP algorithm
    """

    allowed_tcp = {"FAST": FAST}
    """A dict mapping TCP specifiers to implementations (classes)."""

    def __init__(self, env, host, dest, data, delay, tcp, tcp_params):
        self.env = env
        # Flow host
        self._host = host
        # FLow destination
        self._dest = dest
        # Amount of data to transmit (bits)
        self._data = data
        # Time (simulation time) to wait before initial transmission
        self._delay = delay
        # Check for a valid TCP specifier
        if tcp in self.allowed_tcp:
            # Initialize TCP object
            self._tcp = self.allowed_tcp[tcp](self, *tcp_params)
        else:
            raise ValueError("unsupported TCP algorithm \"{}\"".format(tcp))
        # Register this flow with its host, and get its ID
        self._id = self._host.register(self)

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

    def generator(self):
        """Create a packet generator.

        If a data size is given, return a generator which yields
        ceil(data size / packet size) packets.  If data size is None,
        it yields a random, 8-bit number of packets.

        :return: packet generator
        :rtype: generator
        """
        n = 0
        if self._data == None:
            # Pick a random, 8-bit number of packets to send
            n = random.getrandbits(8)
        else:
            # Calculate how many packets are needed to send self._data bits
            n = math.ceil(self._data / resources.Packet.size)

        # Create packet generator
        g = (resources.Packet(self._host.addr, self._dest, self._id, i)
             for i in range(n))

        return g

    def transmit(self, packet):
        """Transmit an outbound packet to localhost.

        :param packet: the data packet to send
        :type packet: :class:`resources.Packet`
        """
        yield self.env.process(self._host.receive(packet))

    def generate(self):
        """Generate packets from this flow."""
        yield self.env.timeout(self._delay)
        yield self.env.process(self._tcp.run())

    def acknowledge(self, ack):
        """Receive an acknowledgement packet.

        If an acknowledgement is received from a packet in the previous
        window, this generator waits <timeout> units of simulation time
        until declaring any unacknowledged packets dropped, and
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
            self._id, self._host.addr, ack.id, self.env.now))
        # Send this acknowledgement to the TCP algorithm
        yield self.env.process(self._tcp.acknowledge(ack))


class Host(object):
    """SimPy process representing a host.

    Each host process has an underlying HostResource which handles
    (de)queueing packets.  This process handles interactions between the
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
            self.addr, packet.flow, packet.id, self.res.env.now))
        # Queue new packet for transmission, and dequeue a packet. The
        # HostResource.receive event returns an outbound ACK if the
        # dequeued packet was an inbound data packet
        packet = yield self.res.receive(packet)

        # Transmit the dequeued packet
        yield self.res.env.process(self.transmit(packet))

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
                                     packet.id, self.res.env.now))
            # Transmit an outbound packet
            yield self.res.env.process(self._transport.send(packet))
        else:
            logger.info("host {} processing ACK {}, {}, {} at time"
                        " {}".format(self.addr, packet.src, packet.flow,
                                     packet.id, self.res.env.now))
            # Send an inbound ACK to its destination flow
            yield self.res.env.process(
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
        return self._link.cost(self._direction)

    def send(self, packet):
        """Send a packet across the link.

        :param packet: the packet to send
        :type packet: :class:`resources.Packet`
        :return: :func:`Link.receive` generating method
        :rtype: generator
        """
        return self._link.receive(self._direction, packet)

    def reverse(self):
        """Return a transport handler in the opposite direction.

        :return: transport handler
        :rtype: :class:`Transport`
        """
        return Transport(self.link, 1 - self.direction)


class Link(object):
    """SimPy process representing a link.

    Each link process has an underlying LinkResource which handles
    (de)queueing packets.  This process handles interactions between the
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
        buffer, and trigger the dequeueing of a packet from that buffer.
        The dequeued packet is then sent across the link.  All link
        buffers are drop-tail.

        :param int direction: the direction to transport the packet
        :param packet: the packet to send through the link
        :type packet: :class:`resources.Packet`
        :return: None
        """
        logger.info("link received packet {}, {}, {} at time {}".format(
            packet.src, packet.flow, packet.id, self.res.env.now))
        # Queue the new packet for transmission, and dequeue a packet
        packets = yield self.res.transport(direction, packet)
        # Transmit the dequeued packets, if any
        yield self.res.env.process(self.transmit(direction, packets))

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
                p.src, p.flow, p.id, self.res.env.now))
            # Increment directional traffic
            self.res.update_traffic(direction, p.size)
            yield simpy.util.start_delayed(self.res.env,
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
        # Set of transports connected to outbound links
        self._links = list()
        # Dictionary mapping outbound links to destinations
        self._routing_table = dict()
        # Dictionary mapping outbound links to destinations (used to build
        # a new routing table before replacing old table)
        self._update_table = dict()
        # Arrival time of most recently processed routing packet
        self._last_arrival = float('inf')
        # Dictionary of links to other routers 
        self._finish_table = dict()
        # Flag indicating if router has converged
        self._converged = False
        # timeout duration used to set routing table to recent update
        self._timeout = 50 #every 0.1s
        # timeout duration used to set frequency of routing table updates
        self._bf_period = 5000 #every 5s

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
        self._links.append(transport)

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
            del self._routing_table[transport]
        except KeyError:
            # If it doesn't exist, do nothing
            pass

    def _route(self, address):
        """Return the correct transport handler for the given address.

        :param int address: the destination address
        :return: the transport handler selected by the routing policy
        :rtype: function
        """
        return self._routing_table[address][0]

    def _transmit(self, packet, transport):
        """Transmit an outbound packet.        
       
        :param packet: the outbound packet     
        :type packet: :class:`resources.Packet`  
        :param transport: the outbound transport handler
        :type transport: :class:`resources.Transport`      
        :return: None      
        """              
        logger.info("router {} transmitting packet {}, {}, {} at time {}".format(
            self.addr, packet.src, packet.flow, packet.id, self.res.env.now))

        # Send the packet      
        yield self.res.env.process(transport.send(packet))

    def receive(self, packet):
        """Receive a packet, and yield a transmission event.

        :param packet:
        :type packet: :class:`resources.Packet`
        :return: None
        """
        logger.info("router {} received packet {}, {}, {} at time {}".format(
            self.addr, packet.src, packet.flow, packet.id, self.res.env.now))

        # Push another packet through the queue
        packet = yield self.res.receive(packet)

        
        # first determine what type of packet it is, then process it 
        if isinstance(packet, resources.Routing):
            yield self.res.env.process(self._handle_routing_packet(packet))
        #handle data packet
        else:
            #get destination address from packet
            dst_addr = packet.dest
            #look up outbound link using the routing table
            transport = self._route(dst_addr)
            #tranmit packet
            yield self.res.env.process(self._transmit(packet, transport))

    def _handle_routing_packet(self, packet):
        if type(packet) == resources.Finish:
            # retrieve trasnport object connecting routers
            transport = packet.data
            # and update the link table to indicate the router on
            # other side of link has converged
            self._finish_table[transport] = True
        # if it's a normal routing packet
        else: 
            #logger.info('processing routing packet')
            # extract data from payload
            host, cost, path, port = packet.data
            # update last arrival time of packet
            self._last_arrival = self.res.env.now
            # if packet has already gone through router, ignore it
            if self._addr in path:
                return     

            # update new routing table if host isn't in table
            if not(host in self._update_table.keys()):
                self._update_table[host] = (port, cost)
                yield self.res.env.process(
                    self._broadcast_packet(host, cost, path, port))

            # update new routing table if there's a more efficient path
            if self._update_table[host][1] > cost:
                self._update_table[host] = (port, cost)
                yield self.res.env.process(
                    self._broadcast_packet(host, cost, path, port))
               
            # after receiving a routing packet, begin update timeout
            yield self.res.env.timeout(self._timeout)

            # check to see if router has reached threshold (for the first time)
            if self.res.env.now >= self._last_arrival + self._timeout and \
                not self._converged:
                # set flag
                self._converged = True
                # and send Finish packets to neighboring routers
                # not connected to hosts
                for t in filter(lambda t: Host not in map(
                                type, t.link.endpoints), self._links):
                    #create reference to outbound port for this router 
                    #on the router recieving the packet
                    transport_ref = t.reverse()
                    new_pkt = resources.Finish(transport_ref)
                    yield self.res.env.process(self._transmit(new_pkt, t))

        # check for one degree of convergence (this router and neighbors)
        if self._converged and all(self._finish_table.values()):
            self._routing_table = self._update_table
            # and reset all variables used for tracking convergence
            self._last_arrvial = float('inf')
            self._converged = False
            self._finish_table = {k: False for k in self._finish_table.keys()}
            self._update_table = dict()

    def _broadcast_packet(self, host, cost, path, port):
        """ Whenever a router updates its routing table, it 
            calls _broadcast_packet() to broadcast 
        """

        # create a new routing packet for each outbound link that's
        # not a host and adjust the cost for each link
        for transport in self._links:
            # don't send routing packets to hosts or to the router
            # that sent the recently received packet.
            if Host in map(type, transport.link.endpoints) or \
                    transport == port:
                continue

            # update packet path and cost
            new_path = path + [self._addr]
            new_cost = cost + transport.cost
            # create reference to outbound port for this router on the 
            # router receiving the packet 
            new_link_ref = transport.reverse()
            # create new packet
            new_pkt = resources.Routing((host, new_cost, new_path, new_link_ref))
            # send the newly created packet
            yield self.res.env.process(self._transmit(new_pkt, transport))

    def begin(self):
        """Periodically update routing tables with Bellman-Ford

        :return: None
        """
        try:
            while True:
                logger.info('Bellman-Ford routing table update')
                # loop through all outbound links
                for transport in self._links:
                    # check for any direct connections to hosts
                    if Host in map(type, transport.link.endpoints):
                        new_path = [self._addr]
                        new_cost = transport.cost
                        new_host = next(filter(lambda e: type(e) == Host,
                                               transport.link.endpoints))
                        host_id = new_host._addr
                        # update routing table for this router
                        self._update_table[host_id] = (transport, new_cost)
                        # get list of transport handlers not connected to hosts
                        for t in filter(
                            lambda t: Host not in map(type, t.link.endpoints), 
                            self._links):
                            new_link_ref = t.reverse()
                            new_cost += t.cost
                            new_pkt = resources.Routing((host_id, new_cost, 
                                               new_path, new_link_ref))
                            yield self.res.env.process(self._transmit(new_pkt, 
                                                                      t))
                yield self.res.env.timeout(self._bf_period)
        except simpy.events.Interrupt:
            pass

#TODO:
#edit docstrings/style 
#edit logger statements
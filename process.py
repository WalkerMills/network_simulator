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
import collections

import resources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCPReno(object):
    """SimPy process representing the flow control using TCP reno.

    TCP reno multiplicative increases the flow window size at the 
    acknowledgement of each packet: slow start phase. When we drop 
    packets or reach a threshold, we drop the window down to half its
    current size and enter the congestion avoidance phase. 

    :param: Flow flow: the flow that is using TCP reno
    :param: int window: initial window size (in packets)
    :param: int timeout: packet acknowledgement timeout
    """
    def __init__(self, flow, window, timeout):
        # The flow running this TCP algorithm
        self._flow = flow
        # window size (bits)
        self._window = window
        # Length of time we wait before packets are considered dropped
        self._timeout = timeout

        self._slow_start = 1

        # Start the packet generator
        self._gen = self._flow._generator()

        # Mapping of unacknowledged packet id -> packet
        self._flow._outbound = dict()

         # Hash table mapping unacknowledged packet -> departure time
        self._unacknowledged = dict()


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

    def _next_packet(self):
        logger.info(
            "flow {}, {} sending packet {} at time {}".format(
                self._flow._id, self._flow._host.addr, self._flow._sent, 
                self._flow._env.now))
        # Generate the next packet to send
        packet = next(self._flow._packets)
        # Increment the count of generated packets
        self._flow._sent += 1
        # Increment the counter for data sent in this window
        self._flow._window2_sent += resources.Packet.size
        # Add the packet to the map of unacknowedged packets
        self._flow._outbound[packet.id] = packet
        # Send the packet through the connected host
        yield self._flow._env.process(self._flow._host.receive(packet))

    def generate(self):
        """Generate and transmit outbound data packets.

        This process has an underlying generator which yields the number
        of packets required to transmit all the data.  The process then
        passivates until the acknowledgement of those packets reactivates
        it.

        :return: None
        """
        # Yield initial delay
        yield self._flow._env.timeout(self._flow._delay)

        # Counter for the number of completed windows
        window_count = 0

        # Send all the packets
        while True:
            try:
                # Send as many packets as fit in our window
                while self._flow._window_sent < self._flow._window:
                    logger.info(
                        "flow {}, {} sending packet {} at time {}".format(
                            self._flow._id, self._flow._host.addr, self._flow._sent, 
                            self._flow._env.now))
                    # Generate the next packet to send
                    packet = next(self._flow._packets)
                    # Increment the count of generated packets
                    self._flow._sent += 1
                    # Increment the counter for data sent in this window
                    self._flow._window_sent += resources.Packet.size
                    # Add the packet to the map of unacknowedged packets
                    self._flow._outbound[packet.id] = packet
                    # Transmit Packet
                    yield self._flow._env.process(self._flow.transmit(packet))
                # Passivate packet generation
                yield self._flow._wait_for_ack

                logger.info(
                        "Finished acknowledging window {} at time {}".format(
                            window_count, self._flow._env.now))

                # On acknowledgements we send the corresponding packet of
                # next window, call by the acknowledge function. Once we have
                # Transmitted all packets in our window we switch to the next 
                # and resume sending packets.

                # On reactivation, reset data counter for new window
                self._flow._window_sent = self._flow._window2_sent
                self._flow._window2_sent = 0
                window_count += 1
            except StopIteration:
                # Stop trying to send packets once we run out
                break
        # Wait for the packets in the last window
        yield self._flow._wait_for_ack
        logger.info("flow {}, {} finished transmitting".format(self._flow._host.addr,
                                                               self._flow._id))

    def retransmit(self):
        """Retransmit all unacknowedged packets. Called when we experience a 
        timeout or have three duplicate acknowledgements. """

        for packet in self._flow._outbound:

            logger.info(
                "flow {}, {} retransmit packet {} at time {}".format(
                    self._flow._id, self._flow._host.addr, packet.id, 
                    self._flow._env.now))
            
            # Transmit Packet
            yield self._flow.env.process(self._flow.transmit(packet))

        # Passivate packet generation
        yield self._wait_for_ack


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
        
        # store ID of packet being acknowledged

        pid = ack.id

        # store ID of next expected package

        expected = ack.data

        # For an acknowledgement received, send packet from next window
        self._next_packet()

        self._detect_duplicate(ack)

        # if the flow is in slow start phase
        if self._slow_start == 1:
            # increase window size exponentially
            self._flow._window += resources.Packet.size

            # enter congestion avoidance if we have reached the ss threshold
            if self._flow._window >= self._flow._ssthresh:
                self._slow_start = 0
        # if the flow is in congestion avoidance phase
        else:
            # increase the window size linearly
            self._flow._window += resources.Packet.size / self._window

        logger.info("flow {}, window size increases to {} at time {}".format(
            self._flow._id, self._flow._window, self._flow._env.now)) 
       
        # Packet Loss Detection #1: Timeout
        timeoutprocess = self._flow._env.process(self._wait_timeout())
        yield timeoutprocess
        # TODO: timeoutprocess.interrupt()

        # if the ACK packet is not from a retransmitted packet
        if ack.id >= self._flow._sent - self._flow._window:
            # Continue sending more packets
            self._flow._wait_for_ack.succeed()
            self._flow._wait_for_ack = self._flow._env.event()
    
    def _wait_timeout(self):
        """ Wait for timeout time for acknowledgement packets

        When a flow completes sending packets of the current window
        and there are packets that have not been acknowledged, 
        :return: None
        """
        # Packet Loss Detection #1: Timeout
        # if we sent enough packets to fill the current window
        # and we have outstanding packets that have not been acknowledged yet
        while self._flow._window_sent >= self._flow._window and len(self._flow._outbound) != 0:
            logger.info("Waiting for timeout at time {}".format(self._flow._env.now))
            # wait for timeout time to see that we receive the ACK
            try:
                yield self._flow._env.timeout(self._timeout)
                # we've timed out, so half the window size
                logger.info("Flow Timeout at time {}".format(self._flow._env.now))
                self._flow._window /= 2
                # TODO: call fast recovery
                # break out of while loop
                break
            except simpy.Interrupt: # condition (len(_outbound) == 0)
                # when we receive an interrupt, we stop waiting
                return

    def _detect_duplicate(self, ack):
        """ Keeps track of duplicate acknowledgements for TCP reno

        If the id of the acknowledgement packet is that of a packet
        that we have not yet acknowledged, remove from the outbound dict.
        If not, the id will not be in the dict, and thus is a duplicate id.
        Increment the duplicate counter 
        """
        # Acknowledge the packet whose ACK was received
        
        if ack.id in self._flow._outbound.keys() and ack.id == ack.expected - 1:
            # Packet was successfully acknowledged when expected 
            # was incremented by one.
            del self._flow._outbound[ack.id]
            self._flow._dupCount = 0
        else:
            # 
            logger.info("We have a duplicate of expected id {}".format(
                ack.expected))
            self._flow._dupCount += 1
       
        if self._flow._dupCount == 3:
            logger.info("Received 3 duplicate ACKS of expected {} at time {}".format(
                ack.id, self._flow._env.now))
            self._flow._window /= 2
            # enter fast retransmit
            yield self._flow.env.process(self.retransmit())

    def run(self):
        """ Run TCP reno & send data

        :return: None
        """
        # Start
        #proc = self._flow._env.process(self._tcp_reno())
        yield self._flow._env.process(self.generate())

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
    :param int timeout: the (simulation) time to wait for :class:`ACK`'s 
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
        # The flow is either is SS(0) or CA(1) states
        self._state = 0
        # The slowstart threshold. Set it to be a large number
        self._ssthresh = 100000
        # Queue to check for duplicate ACKs for packet loss
        self._dupQueue = collections.deque(maxlen = 4)
        # Counter to check for duplicate ACKs for packet loss
        self._dupCount = 0
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
        # Initialize window sent data counter for the next window
        self._window2_sent = 0
        # Wait to receive acknowledgement (passivation event)
        self._wait_for_ack = self._env.event()

        # Create the TCP class we wish to use
        self._tcp = TCPReno(self, self._window, self._time)

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
        """Generate packets from this flow."""
        yield self._env.timeout(self._delay)
        yield self._env.process(self._tcp.run())

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

        # call the TCP we want this flow to use
      
        yield self._env.process(self._tcp.acknowledge(ack))


    def transmit(self, packet):
        """Transmit an outbound packet to localhost.
        :param packet: the data packet to send
        :type packet: :class:`resources.Packet`
        """
        yield self._env.process(self._host.receive(packet))

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

        :param function transport: transport handler
        :return: None
        """
        self._transport = transport

    def disconnect(self, transport):
        """Disconnect an existing transport handler.

        If the given transport handler is not connected to this host,
        do nothing.

        :param function transport: the transport handler to disconnect
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
            yield self.res._env.process(self._transport(packet))
        else:
            logger.info("host {} transmitting inbound ACK {}, {}, {} at time"
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
        self._up = lambda p: self.receive(resources.UP, p)
        # "Download" handler
        self._down = lambda p: self.receive(resources.DOWN, p)

    @property
    def delay(self):
        """The link delay in seconds."""
        return self._delay

    @property
    def endpoints(self):
        """A list of connected endpoints (up to 1 per direction)"""
        return self._endpoints

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

        :param function transport: transport handler
        :return: None
        """
        # Connect a new transport handler to a new "port"
        self._links.add(transport)

    def disconnect(self, transport):
        """Disconnect a link, if it exists.

        :param function transport: the transport handler to disconnect
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
        yield self.res._env.process(transport(packet))
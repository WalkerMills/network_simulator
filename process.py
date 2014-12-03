"""
.. module:: process
    :platform: Unix
    :synopsis: This module defines network actors as processes
"""

import itertools
import logging
import math
import random
import simpy
import collections

from collections import deque
from queue import Queue, Empty

import resources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAST(object):
    """SimPy process implementing FAST TCP.

    :param flow: the flow to link this TCP instance to
    :type flow: :class:`Flow`
    :param int window: initial window size (in packets)
    :param int timeout: packet acknowledgement timeout (simulation time)
    :param int alpha: desired enqueued data at equilibrium (bits)
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
        self._alpha = alpha / resources.Packet.size
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
        self._dropped = Queue()
        # Next ID's expected by the destination host.  Last 4 are cached to
        # check for packet dropping (3 duplicate ACK's)
        self._next = deque(maxlen=4)
        # Window updating process
        self._window_ctrl_proc = None
        # Finish event
        self._finished = simpy.events.Event(self._flow.env)

    @property
    def finished(self):
        """Transmission finished event.

        This event is only successful once all data packets have been
        sent, and any dropped packets retransmitted successfully.
        """
        return self._finished

    @property
    def timeout(self):
        """The time after which packets are considered dropped.

        :return: acknowledgement timeout (simulation time)
        :rtype: int
        """
        return self._timeout

    @property
    def window(self):
        """Transmission window size, in bits.

        :return: window size (bits)
        :rtype: int
        """
        return self._window

    def _estimate(self, trip):
        """Update the mean round trip time & estimated queueing delay.

        Mean round trip time is calculated as a moving average, and
        queueing delay is taken to be the difference between that mean
        and the minimum observed round trip time.
        """
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
        """Append timed out packets to the dropped packet queue."""
        # For each unacknowledged packet
        for packet, time in list(self._unacknowledged.items())[:]:
            # If it has timed out
            if time <= self._flow.env.now - self._timeout:
                # Mark it as dropped
                self._dropped.put(packet)
                # Remove it from the unacknowledged packets
                # del self._unacknowledged[packet]

    def _update_window(self):
        """Update the maximum window size."""
        # Calculate the new window size, as directed by FAST TCP spec
        window = (1 - self._gamma) * self._window + self._gamma * \
            (self._window * self._min_trip / self._mean_trip + self._alpha)
        # Update window size by at most doubling it
        self._window = min(2 * self._window, window)
        print("\n{}\n".format(self._window))

    def _window_control(self):
        """Periodically update the window size."""
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

    def _window_size(self):
        """Calculate the transimssion window size.

        Window size is equal to the difference in the ID's of the earliest
        and latest packets sent, plus one.
        """
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
        packet = next(filter(lambda p: p.id == pid, 
                      self._unacknowledged.keys()))
        # Calculate round trip time
        rtt = self._flow.env.now - self._unacknowledged[packet]
        # Mark packet as acknowledged
        del self._unacknowledged[packet]
        # Update mean round trip time & queueing delay
        self._estimate(rtt)
        # If we have 3 dropped ACK's
        if self._next.count(self._next[0]) == self._next.maxlen:
            # Enter recovery
            yield self._flow.env.process(self.recover())
        # If there are more packets to generate
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
            gen = (self._dropped.get_nowait() for i in range(
                    self._dropped.qsize()))
            # Update the window (size)
            self._update_window()
        else:
            # Otherwise, take packets from the data packet generator
            gen = self._gen
        try:
            # While the window is not full
            while self._window_size() < int(self._window):
                # Get the next data packet
                packet = next(gen)
                # Transmit the packet
                yield self._flow.env.process(self.transmit(packet))
                # If we are in recovery mode
                if recover:
                    # Update the window (size)
                    self._update_window()
        # Once we have exhausted the packet generator
        except StopIteration:
            # If we exhausted the data packet generator
            if not recover:
                # Set the data packet generator to None
                self._gen = None
                # Wait for acknowledgements in the last window
                yield self._flow.env.timeout(self._timeout)
                # While there remain dropped packets
                while not self._dropped.empty():
                    # Retransmit as many dropped packets as possible
                    yield self._flow.env.process(self.recover())
                    # Wait for acknowledgements
                    yield self._flow.env.timeout(self._timeout)
                    # Update dropped packets
                    self._update_dropped()
                try:
                    # Kill the window control process
                    self._window_ctrl_proc.interrupt()
                except RuntimeError:
                    pass
                # Mark this TCP algorithm as finished if it isn't already
                if not self._finished.triggered:
                    self._finished.succeed()
        except Empty:
            if self._gen is None and not self._finished.triggered:
                self._finished.succeed()

    def get_departure(self, pid):
        """Returns the departure time of a packet with given pid

        :return: None, if packet id doens't match
        :return: time, if packet id has a match
        """
        try:
            _, time = next(filter(lambda p: p[0].id == pid, 
                                  self._unacknowledged.items()))
        except StopIteration:
            time = None
        return time

    def recover(self):
        """Enter recovery mode.

        :return: None
        """
        # Halve the window size
        self._window /= 2
        if self._window < 1:
            self._window = 1
        # Retransmit as many dropped packets as possible
        yield self._flow.env.process(self.burst(recover=True))

    def send(self):
        """Send all data packets.

        :return: None
        """
        # Start the window control process
        self._window_ctrl_proc = self._flow.env.process(self._window_control())
        # Yield a burst process
        yield self._flow.env.process(self.burst())

    def transmit(self, packet):
        """Transmit a data packet to the host flow.

        :param packet: the data packet to send
        :type packet: :class:`resources.Packet`
        :return: None
        """
        # Mark the packet as unacknowledged
        self._unacknowledged[packet] = self._flow.env.now
        # Transmit the packet
        yield self._flow.env.process(self._flow.transmit(packet))


class Reno(object):
    """SimPy process implementing TCP Reno.

    :param flow: the flow that is using TCP Reno
    :type flow: :class:`Flow`
    :param int window: initial window size (in packets)
    :param int timeout: packet acknowledgement timeout
    """

    def __init__(self, flow, window, timeout):
        # The flow running this TCP algorithm
        self._flow = flow
        # window size (packets)
        self._window = window
        # Length of time we wait before packets are considered dropped
        self._timeout = timeout

        # Maximum timeout (64 sec)
        self._max_time = 64000000000
        # Slow start threshold
        self._slow_start = 32
        # Packet generator
        self._gen = self._flow.generator()
        # Hash table mapping unacknowledged packet -> departure time
        self._unacknowledged = dict()
        # Next ID's expected by the destination host.  Last 4 are cached to
        # check for packet dropping (3 duplicate ACK's)
        self._next = deque(maxlen=4)
        # Wait for acknowledgement process
        self._wait_proc = None
        # Recovery mode flag
        self._recovery = False
        # Finish event
        self._finished = simpy.events.Event(self._flow.env)

    @property
    def finished(self):
        """Transmission finished event.

        This event is only successful once all data packets have been
        sent, and any dropped packets retransmitted successfully.
        """
        return self._finished

    @property
    def timeout(self):
        """The time after which packets are considered dropped.

        :return: acknowledgement timeout (simulation time)
        :rtype: int
        """
        return self._timeout

    @property
    def window(self):
        """Transmission window size, in packets.

        :return: window size (packets)
        :rtype: int
        """
        return self._window

    def _wait(self):
        """Wait for acknowledgements; time out if none are recieved."""
        try:
            # Wait for acknowledgement(s)
            yield self._flow.env.timeout(self._timeout)
            # For each unacknowledged packet
            for packet, time in list(self._unacknowledged.items())[:]:
                # If it has timed out
                if time <= self._flow.env.now - self._timeout:
                    # Immediately retransmit the packet
                    self.transmit(packet)
            # Set the slow start threshold to half the window size
            self._slow_start = self._window / 2
            if self._slow_start == 0:
                self._slow_start = 1
            # Reset window size
            self._window = 1
            # Double timeout length
            if self._timeout < self._max_time:
                self._timeout *= 2
        except simpy.Interrupt:
            self._wait_proc = None

    def _window_size(self):
        """Calculate the transimssion window size.

        Window size is equal to the difference in the ID's of the earliest
        and latest packets sent, plus one.
        """
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
        packet = next(filter(lambda p: p.id == pid,
                      self._unacknowledged.keys()))
        # Mark the packet as acknowledged
        del self._unacknowledged[packet]
        # If we just finished fast recovery
        if self._recovery:
            # And we received another duplicate acknowledgement
            if expected == self._next[-1]:
                # Increase window size by one
                self._window += 1
            else:
                # Set the window size to the slow start threshold
                self._window = self._slow_start
        # If we have 3 duplicate ACK's
        if self._next.count(self._next[0]) == self._next.maxlen:
            # Get the dropped packet
            dropped = next(filter(lambda p: p.id == expected,
                           self._unacknowledged.keys()))
            # Immediately retransmit the dropped packet
            yield self._flow.env.process(self.transmit(dropped))
            # Enter fast recovery mode
            self._slow_start = self._window / 2
            self._window = self._slow_start + 3
            self._recovery = True
        else:
            # If in slow start mode
            if self._window_size() <= self._slow_start:
                # Increase window size by one packet
                self._window += 1
            else:
                # Otherwise, linearly increase the window size
                self._window += resources.Packet.size / self._window
        # Stop the waiting process, if any
        if self._wait_proc is not None:
            self._wait_proc.interrupt()

    def get_departure(self, pid):
        """Returns the departure time of a packet with given pid.

        If there is a match for the packet id, return its departure time,
        otherwise, return None

        :return: departure time
        :rtype: int or None
        """
        try:
            _, time = next(filter(lambda p: p[0].id == pid, 
                                  self._unacknowledged.items()))
        except StopIteration:
            time = None
        return time

    def send(self):
        """Send all data packets.

        :return: None
        """
        try:
            while True:
                # While we have room in the window
                while self._window_size() < self._window:
                    # Get the next data packet
                    packet = next(self._gen)
                    # Transmit the data packet
                    yield self._flow.env.process(self.transmit(packet))
                # Wait for acknowledgements, or timeout
                self._wait_proc = self._flow.env.process(self._wait())
                yield self._wait_proc
        except StopIteration:
            # Wait for packets in the last window
            self._wait_proc = self._flow.env.process(self._wait())
            yield self._wait_proc
            # If there remain unacknowledged packets
            if len(self._unacknowledged) > 0:
                # Make a generator for the dropped packets
                self._gen = (p for p in list(self._unacknowledged.keys())[:])
                # Resend all dropped packets
                yield self._flow.env.process(self.send())
            # Mark this TCP algorithm as finished, if it isn't already
            if not self._finished.triggered:
                self._finished.succeed()

    def transmit(self, packet):
        """Transmit a data packet to the host flow.

        :param packet: the data packet to send
        :type packet: :class:`resources.Packet`
        :return: None
        """
        # Mark the packet as unacknowledged
        self._unacknowledged[packet] = self._flow.env.now
        # Transmit the packet
        yield self._flow.env.process(self._flow.transmit(packet))


class Flow(object):
    """SimPy process representing a flow.

    Each flow process is connected to a source host, and generates as
    many packets as are necessary to send all of its data.  If data is
    None, a random, 8-bit number of packets are sent.  Every flow needs
    a TCP algorithm to be specified, from among the currently supported
    TCP algorithms.  At the moment, the only allowed specifiers are 
    \"FAST\", and \"Reno\".  See :class:`FAST` or :class:`Reno` for 
    details on what tcp_params should look like.

    :param simpy.Environment env: the simulation environment
    :param host: the source host of this flow
    :type host: :class:`Host`
    :param int dest: the address of the destination host
    :param int data: the total amount of data to transmit (bits)
    :param int delay: the simulation time to wait before sending any packets
    :param str tcp: TCP algorithm specifier
    :param list tcp_params: parameters for the TCP algorithm
    """

    allowed_tcp = {"FAST": FAST, "Reno": Reno}
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
        # Bits transmitted by flow
        self._transmitted = 0
        # Bits received by flow
        self._received = 0
        # List of roundtrip times for packets in given timestep
        self._times = list()
        # Reference time for collecting packet RTT's
        self._last_arrival = 0
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
    def finished(self):
        """Transmission finished event.

        This event is only successful once all data packets have been
        sent, and any dropped packets retransmitted successfully.

        :return: finished event
        :rtype: ``simpy.events.Event``
        """
        return self._tcp.finished

    @property
    def host(self):
        """The host of this flow.

        :return: localhost
        :rtype: :class:`Host`
        """
        return self._host

    @property
    def id(self):
        """The ID of this flow, assigned by its host.

        :return: flow id, relative to localhost
        :rtype: int
        """
        return self._id

    def _update_rtt(self, pid):
        """Update list of packet RTT's."""
        # Get departure time of packet
        depart = self._tcp.get_departure(pid)
        if depart is None:
            return
        # Calcualte roundtrip time of packet
        rtt = self.env.now - depart

        # if this is the first packet in a new timestep,
        if self.env.now  > self._last_arrival:
            # reset list of rtt's
            self._times = list()
            # update latest arrival time
            self._last_arrival = self.env.now
        # append packet to list of RTT's
        self._times.append(rtt)
        self.env.update(
            "Round trip times,{},{}".format(self._host.addr, self._id),
            sum(self._times) / len(self._times))

    def acknowledge(self, ack):
        """Receive an acknowledgement packet.

        Acknowledgement packets are passed to the underlying TCP algorithm
        to be handled.

        :param ack: acknowledgement packet
        :type ack: :class:`resources.ACK`
        :return: None
        """
        #logger.info("flow {}, {} acknowledges packet {} at time {}".format(
        #    self._id, self._host.addr, ack.id, self.env.now))

        # Update round trip times
        self._update_rtt(ack.id)
        # update number of received bits
        self._received += resources.ACK.size
        self.env.update(
            "Flow received,{},{}".format(self._host.addr, self._id),
            self._received)

        # Send this acknowledgement to the TCP algorithm
        yield self.env.process(self._tcp.acknowledge(ack))

    def generate(self):
        """Generate packets from this flow."""
        yield self.env.timeout(self._delay)
        yield self.env.process(self._tcp.send())

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
        g = (resources.Packet(self._host.addr, self.dest, self.id, i)
             for i in range(n))

        return g

    def transmit(self, packet):
        """Transmit an outbound packet to localhost.
        
        :param packet: the data packet to send
        :type packet: :class:`resources.Packet`
        """
        #update transmitted number of bits
        self._transmitted += packet.size
        self.env.update(
            "Flow transmitted,{},{}".format(self._host.addr, self.id),
            self._transmitted)
        # Transmit the packet
        yield self.env.process(self._host.receive(packet))
        # Wait packet size / link capacity before terminating
        yield self.env.timeout(
            packet.size * 1e9 / self._host.transport.capacity)


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
        # Bits transmitted by host
        self._transmitted = 0
        # Bits received by host
        self._received = 0

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

    @property
    def transport(self):
        """The connected transport handler, if it exists.

        :return: transport handler
        :rtype: :class:`Transport` or None
        """
        return self._transport

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
        if packet.dest == self.addr:
            # update bits received by host
            self._received += packet.size
            # create getter for received data
            self.res.env.update("Host received,{}".format(self.addr),
                                self._received)

        # Queue new packet for transmission, and dequeue a packet. The
        # HostResource.receive event returns an outbound ACK if the
        # dequeued packet was an inbound data packet
        packet = yield self.res.receive(packet)

        # Transmit the dequeued packet
        yield self.res.env.process(self.transmit(packet))

    def register(self, flow):
        """Register a new flow on this host, and return the flow ID.

        :param flow: a new flow to register with this host
        :type flow: :class:`Flow`
        :return: None
        """
        self._flows.append(flow)
        return len(self._flows) - 1

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
            #logger.info("host {} transmitting packet {}, {}, {} at time"
            #            " {}".format(self.addr, packet.src, packet.flow, 
            #                         packet.id, self.res.env.now))
            
            # update bits transmitted by host
            self._transmitted += packet.size
            self.res.env.update("Host transmitted,{}".format(self._addr),
                                self._transmitted)
            # Transmit an outbound packet
            yield self.res.env.process(self._transport.send(packet))
        else:
            #logger.info("host {} processing ACK {}, {}, {} at time"
            #            " {}".format(self.addr, packet.src, packet.flow,
            #                         packet.id, self.res.env.now))
            # Send an inbound ACK to its destination flow
            yield self.res.env.process(
                self._flows[packet.flow].acknowledge(packet))


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

    def __init__(self, env, capacity, size, delay, addr):
        # Initialize link buffers
        self.res = resources.LinkBuffer(env, size, addr)
        # Link capacity (bps)
        self._capacity = capacity
        # Link delay (simulation time)
        self._delay = delay
        # link address
        self._addr = addr

        # Endpoints for each direction
        self._endpoints = [None, None]
        # "Upload" handler
        self._up = Transport(self, resources.UP)
        # "Download" handler
        self._down = Transport(self, resources.DOWN)
        # Link traffic (bps)
        self._traffic = [0, 0]
        # Total number of bits transmitted by link
        self._transmitted = 0
        # Buffer flushing processes
        self._flush_proc = (self.res.env.process(self._flush(resources.UP)),
                            self.res.env.process(self._flush(resources.DOWN)))

    @property
    def capacity(self):
        """The maximum bitrate of the link in bps."""
        return self._capacity

    @property
    def delay(self):
        """The link delay in seconds."""
        return self._delay

    @property
    def endpoints(self):
        """A list of connected endpoints (up to 1 per direction)"""
        return self._endpoints

    def _available(self, direction):
        """The available link capacity in the given direction.

        :param int direction: link direction
        :return: directional traffic (bps)
        :rtype: int
        """
        return self._capacity - self._traffic[direction]

    def _flush(self, direction):
        while True:
            yield self.res.env.timeout(
                self.res.last_size[direction] * 1e9 / self._capacity)
            if resources.Packet.size <= self._available(direction):
                packet = self.res.dequeue(direction)
                if packet is not None:
                    self.res.env.process(self._transmit(direction, packet))

    def _transmit(self, direction, packet):
        """Transmit a packet across the link in a given direction.

        This generating function yields a transmission process for each
        packet given

        :param int direction: the direction to transport the packet
        :param packet: a packet to transmit
        :type packet: :class:`resources.Packet`
        :return: None
        """
        #logger.info("transmitting packet {}, {}, {} at time {}".format(
        #    packet.src, packet.flow, packet.id, self.res.env.now))
        # Update total bits transmitted by link
        self._transmitted += packet.size
        # create getter for transmitted data
        self.res.env.update("Link transmitted,{}".format(self.res.addr),
                            self._transmitted)
        # update average of buffer fill
        self.res.update_buffered(direction, self.res.env.now)
        # Transmit packet after waiting, as if sending the packet
        # across a physical link
        yield simpy.util.start_delayed(self.res.env,
            self._endpoints[direction].receive(packet), self._delay)

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

    def cost(self, direction):
        """Return the cost of a direction on this link.

        Total cost is calculated as the product of propagation delay,
        (data) packet size, and 1 + buffer fill proportion, divided by
        the logarithm of available capacity.

        :param int direction: link direction to compute cost for

        """
        logger.info("L{} --> cost:{}".format(self._addr, 
            self._delay + self.res.buffered(direction)))
        return self._delay + self.res.buffered(direction)

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
        buffer.  All link buffers are drop-tail.

        :param int direction: the direction to transport the packet
        :param packet: the packet to send through the link
        :type packet: :class:`resources.Packet`
        :return: None
        """
        #logger.info("link{} received packet {}, {}, {} at time {}".format(
        #    self._addr, packet.src, packet.flow, packet.id, self.res.env.now))
        # Enqueue the new packet, and check if the packet wasn't dropped
        dropped = yield self.res.enqueue(direction, packet)


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
        # Dictionary mapping destination -> (transport, path cost)
        self._routing_table = dict()
        # Dictionary used to build a new routing table during an update
        self._update_table = dict()
        # Dictionary used to assess local convergence of routing tables
        self._finish_table = dict()
        # Arrival time of most recently processed routing packet
        self._last_arrival = float('inf')
        # Flag indicating if router has converged
        self._converged = False
        # timeout duration used to set routing table to recent update
        self._timeout = 50000000 #every 0.1s
        # timeout duration used to set frequency of routing table updates
        self._bf_period = 500000000 #every 5s

    @property
    def addr(self):
        """The address of this router.

        :return: router address
        :rtype: int
        """
        return self._addr

    def _broadcast_packet(self, host, cost, path, rec_port):
        """Broadcast a routing packet."""

        # create a new routing packet for each outbound link that's
        # not a host and adjust the cost for each link
        for transport in self._links:
            # don't send routing packets to hosts or to the router
            # that sent the recently received packet.
            if Host in map(type, transport.link.endpoints) or \
                transport == rec_port:
                continue

            # update packet path and cost
            #new_path = path + [self._addr]
            #new_cost = cost + transport.cost
            # create reference to outbound port for this router on the 
            # router receiving the packet 
            #new_port = transport.reverse()
            # create new packet
            new_pkt = resources.Routing((host, cost + transport.cost, 
                path + [self._addr], transport.reverse()))
            # send the newly created packet
            logger.info("R{} broadcasting on L{}".format(self._addr,
                        transport.link._addr))
            yield self.res.env.process(self.transmit(new_pkt, transport))

    def _handle_routing_packet(self, packet):
        """Handle a routing or finish packet."""
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
            host, cost, path, rec_port = packet.data
            # update last arrival time of packet
            self._last_arrival = self.res.env.now
            # if packet has already gone through router, ignore it
            if self._addr in path:
                return     

            # update new routing table if host isn't in table
            if not (host in self._update_table.keys()):
                self._update_table[host] = (rec_port, cost)
                logger.info("R{} create: L{}, cost {}, H{}".format(
                    self._addr, rec_port.link._addr, cost, host))
                yield self.res.env.process(
                    self._broadcast_packet(host, cost, path, rec_port))

            # update new routing table if there's a more efficient path
            if self._update_table[host][1] > cost:
                logger.info("R{} replace H{}: L{}, cost {} to L{}, cost {}".format(
                    self._addr, host, self._update_table[host][0].link._addr, 
                    self._update_table[host][1], rec_port.link._addr, cost))
                self._update_table[host] = (rec_port, cost)
                yield self.res.env.process(
                    self._broadcast_packet(host, cost, path, rec_port))
               
            # after receiving a routing packet, begin update timeout
            yield self.res.env.timeout(self._timeout)

            # check to see if router has reached threshold (for the first time)
            if self.res.env.now >= self._last_arrival + self._timeout and \
                not self._converged:
                # set flag
                self._converged = True
                logger.info("router {} self converged".format(self._addr))
                # and send Finish packets to neighboring routers
                # not connected to hosts
                for t in filter(
                    lambda t: Host not in map(type, t.link.endpoints),
                    self._links):
                    #create reference to outbound port for this router 
                    #on the router recieving the packet
                    transport_ref = t.reverse()
                    new_pkt = resources.Finish(transport_ref)
                    yield self.res.env.process(self.transmit(new_pkt, t))

        # check for one degree of convergence (this router and neighbors)
        if self._converged and all(self._finish_table.values()):
            # If there is an updated routing table
            logger.info("router {} fully converged".format(self._addr))
            if self._update_table:
                # Replace the old table
                self._routing_table = self._update_table
            # and reset all variables used for tracking convergence
            self._last_arrvial = float('inf')
            self._converged = False
            self._finish_table = dict()
            self._update_table = dict()

    def _route(self, address):
        """Return the correct transport handler for the given address.

        :param int address: the destination address
        :return: the transport handler selected by the routing policy
        :rtype: function
        """
        return self._routing_table[address][0]

    def begin(self):
        """Periodically update routing tables with Bellman-Ford.

        :return: None
        """
        try:
            while True:
                # loop through all outbound links
                for transport in self._links:
                    # check for any direct connections to hosts
                    if Host in map(type, transport.link.endpoints):
                        logger.info('Bellman-Ford routing table update')
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
                            new_port = t.reverse()
                            #new_cost += t.cost
                            new_pkt = resources.Routing(
                                (host_id, new_cost + t.cost, new_path, new_port))
                            yield self.res.env.process(
                                self.transmit(new_pkt, t))
                    # initialize finish_table for neighboring routers
                    else:
                        self._finish_table[transport] = False
                yield self.res.env.timeout(self._bf_period)
        except simpy.events.Interrupt:
            pass

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
        try:
            # Disconnect the given transport handler
            self._links.remove(transport)
        except ValueError:
            # If it doesn't exist, do nothing
            pass
        # Remove any routing entries using the given transport handler
        self._routing_table = \
            {k:v for k, v in self._routing_table if v[0] != transport}
        self._update_table = \
            {k:v for k, v in self._update_table if v[0] != transport}

    def receive(self, packet):
        """Receive a packet, and yield a transmission event.

        :param packet:
        :type packet: :class:`resources.Packet`
        :return: None
        """
        #logger.info("router {} received packet {}, {}, {} at time {}".format(
        #    self.addr, packet.src, packet.flow, packet.id, self.res.env.now))

        # Push another packet through the queue
        packet = yield self.res.receive(packet)
        # first determine what type of packet it is, then process it 
        if isinstance(packet, resources.Routing):
            yield self.res.env.process(self._handle_routing_packet(packet))
        # handle data packet
        else:
            # look up outbound link using the routing table
            transport = self._route(packet.dest)
            # tranmit packet
            yield self.res.env.process(self.transmit(packet, transport))
            # Wait before sending another packet
            #yield self.res.env.timeout(packet.size * 1e9 / transport.capacity)

    def transmit(self, packet, transport):
        """Transmit an outbound packet.        
       
        :param packet: the outbound packet     
        :type packet: :class:`resources.Packet`  
        :param transport: the outbound transport handler
        :type transport: :class:`resources.Transport`      
        :return: None      
        """              
        #logger.info("router {} transmitting packet {}, {}, {} at time "
        #            "{}".format(self.addr, packet.src, packet.flow, packet.id,
        #                        self.res.env.now))
        # Send the packet      
        yield self.res.env.process(transport.send(packet))

#TODO:
#edit docstrings/style 
#edit logger statements


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
    def capacity(self):
        """Capacity of the underlying link."""
        return self._link.capacity

    @property
    def cost(self):
        """Directional cost of the underlying link."""
        return self._link.cost(self._direction)

    @property
    def direction(self):
        """The direction of this transport handler."""
        return self._direction

    @property
    def link(self):
        """The link that owns this transport handler."""
        return self._link

    def reverse(self):
        """Return a transport handler in the opposite direction.

        :return: transport handler
        :rtype: :class:`Transport`
        """
        return Transport(self.link, 1 - self.direction)

    def send(self, packet):
        """Send a packet across the link.

        :param packet: the packet to send
        :type packet: :class:`resources.Packet`
        :return: :func:`Link.receive` generating method
        :rtype: generator
        """
        return self._link.receive(self._direction, packet)

"""
.. module:: process
    :platform: Unix
    :synopsis: This module defines network actors as processes
"""

import abc
import itertools
import math
import queue
import random
import simpy

from collections import deque

import resources


class TCP(metaclass=abc.ABCMeta):
    """TCP algorithm abstract base class.

    This class defines a number of properties & utility functions for
    use by subclasses, as well as a mostly abstract API for use by the
    :class:`Flow` class.

    :param flow: the flow to link this TCP instance to
    :type flow: :class:`Flow`
    :param int window: initial window size (packets)
    :param int timeout: packet acknowledgement timeout (ns)
    """

    def __init__(self, flow, window, timeout):
        # The flow running this TCP algorithm
        self._flow = flow
        # Window size (packets)
        self._window = window
        # Time after which packets are considered dropped
        self._timeout = timeout

        # Packet generator
        self._gen = self._flow.generator()
        # Hash table mapping unacknowledged packet -> departure time
        self._unacknowledged = dict()
        # Maximum timeout (64 sec)
        self._max_time = 64000000000
        # Finish event
        self._finished = simpy.events.Event(self._flow.env)

    @property
    def finished(self):
        """Transmission finished event.

        This event is only successful once all data packets have been
        sent, and any dropped packets retransmitted successfully.

        :return: finish event
        :rtype: ``simpy.events.Event``
        """
        return self._finished

    @property
    def flow(self):
        """The flow that this TCP algorithm is connected to.

        :return: host flow
        :rtype: :class:`Flow`
        """
        return self._flow

    @property
    def max_time(self):
        """The maximum acknowledgement timeout length.

        :return: maximum timeout (ns)
        :rtype: int
        """
        return self._max_time

    @property
    def timed_out(self):
        """List of timed out packets.

        :return: timed out packets
        :rtype: [:class:`resources.Packet`]
        """
        return [p for p, t in self._unacknowledged.items() 
                if t <= self.flow.env.now - self._timeout]

    @property
    def timeout(self):
        """The time after which packets are considered dropped.

        :return: acknowledgement timeout (ns)
        :rtype: int
        """
        return self._timeout

    @timeout.setter
    def timeout(self, timeout):
        """Set the acknowledgement timeout.

        Timeouts less than 1 are not accepted, and will set the timeout
        to 1.

        :param int timeout:
        :return: None
        """
        self._timeout = max(timeout, 1)

    @property
    def unacknowledged(self):
        """Unacknowledged packet dictionary.

        This dictionary maps unackowledged packets to their departure time.

        :return: unackowledged packets
        :rtype: {:class:`resources.Packet`: int}
        """
        return self._unacknowledged

    @property
    def window(self):
        """Transmission window size (bits).

        :return: window size (bits)
        :rtype: int
        """
        return self._window

    @window.setter
    def window(self, window):
        """Set the window size.

        Window sizes less than 1 are not accepted, and will set the window
        to 1.

        :param int window: new window size
        :return: None
        """
        self._window = max(window, 1)

    @abc.abstractmethod
    def acknowledge(self, ack):
        """Process an acknowledgement packet.

        This abstract method must be implemented by a subclass.

        :param ack: the acknowledgement packet
        :type ack: :class:`resources.ACK`
        :return: None
        """
        pass

    def departure(self, pid):
        """Get the departure time of an unacknowledged packet.

        Raises a KeyError if the packet is not currently unacknowledged.

        :return: time, if packet id has a match
        :rtype: int or None
        """
        try:
            # Retrieve the departure time of the packet with the given id
            _, time = next(filter(lambda i: i[0].id == pid, 
                                  self._unacknowledged.items()))
            return time
        except StopIteration:
            raise KeyError("packet {} is not unacknowledged".format(pid))

    def mark(self, pid):
        """Mark a packet as acknowledged.

        Raises a KeyError if the packet is not currently unacknowledged.

        :param int pid: the id of the packet to acknowledge
        :return: None
        """
        del self._unacknowledged[self.packet(pid)]

    def packet(self, pid):
        """Get an unacknowledged packet with the given id.

        Raises a KeyError if the packet is not currently unacknowledged.

        :param int pid: packet id
        :return: the unacknowledged packet
        :rtype: :class:`resources.Packet`
        """
        try:
            packet = next(filter(lambda p: p.id == pid, 
                                 self._unacknowledged.keys()))
            return packet
        except StopIteration:
            raise KeyError("packet {} is not unacknowledged".format(pid))

    @abc.abstractmethod
    def send(self):
        """Send all data packets.

        This abstract method must be implemented by a subclass.

        :return: None
        """
        pass

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


class FAST(TCP):
    """SimPy process implementing FAST TCP.

    :param flow: the flow to link this TCP instance to
    :type flow: :class:`Flow`
    :param int window: initial window size (packets)
    :param int timeout: packet acknowledgement timeout (ns)
    :param int alpha: desired number of enqueued packets at equilibrium
    :param float gamma: window update weight
    """

    def __init__(self, flow, window, timeout, alpha, gamma=0.5):
        super(FAST, self).__init__(flow, window, timeout)
        # Packets in buffer at equilibrium
        self._alpha = alpha
        # Window update weight
        self._gamma = gamma

        # Mean round trip time
        self._mean_trip = float("inf")
        # Minimum observed round trip time
        self._min_trip = float("inf")
        # Estimated queuing delay
        self._delay = 0
        # Next ID's expected by the destination host.  Last 4 are cached to
        # check for packet dropping (3 duplicate ACK's)
        self._next = deque(maxlen=4)
        # Window updating process
        self._window_ctrl_proc = None

    def _estimate(self, trip):
        """Update the mean round trip time & estimated queuing delay.

        Mean round trip time is calculated as a moving average, and
        queuing delay is taken to be the difference between that mean
        and the minimum observed round trip time.
        """
        # Calculate weight for round trip mean update
        eta = min(3.0 / self.window, 0.25)
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
        # Update estimated queuing delay
        self._delay = self._mean_trip - self._min_trip

    def _update_window(self):
        """Update the maximum window size."""
        # Calculate the new window size, as directed by FAST TCP spec
        window = (1 - self._gamma) * self.window + self._gamma * \
            (self.window * self._min_trip / self._mean_trip + self._alpha)
        # Update window size by at most doubling it
        self.window = min(2 * self.window, window)
        # Update monitored window size
        self.flow.env.update(
            "Window size,{},{},{}".format(self.flow.host.addr, self.flow.dest,
                                          self.flow.id), 
            self._window)

    def _window_control(self):
        """Periodically update the window size."""
        try:
            while True:
                # Wait for acknowledgements
                yield self.flow.env.timeout(self._timeout)
                # If there exist timed out packets
                if self.timed_out:
                    # Enter recovery mode (updates window size)
                    yield self.flow.env.process(self.recover())
                else:
                    # Otherwise, just update the window size
                    self._update_window()
        except simpy.events.Interrupt:
            pass

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
        # Calculate round trip time
        rtt = self.flow.env.now - self.departure(pid)
        # If 2 * RTT is greater than the current timeout, update it
        self.timeout = min(max(self.timeout, 2 * rtt), self.max_time)
        # Update mean round trip time & queuing delay
        self._estimate(rtt)
        # Mark packet as acknowledged
        self.mark(pid)
        # If we have 3 dropped ACK's
        if self._next.count(self._next[0]) == self._next.maxlen:
            # Enter recovery
            yield self.flow.env.process(self.recover())
        # If there are more packets to generate
        if self._gen is not None:
            # Try to inject packets into the network
            yield self.flow.env.process(self.burst())

    def burst(self, recover=False):
        """Inject packets into the network.

        If recover is True, then the packets are taken from the set of
        timed out packets.

        :param bool recover: flag for recovery mode
        :return: None
        """
        # If we are in recovery mode
        if recover:
            # Create a dropped packet generator
            gen = (p for p in self.timed_out[:])
        else:
            # Otherwise, take packets from the data packet generator
            gen = self._gen
        try:
            # While the window is not full
            while len(self.unacknowledged) < int(self.window):
                # Get the next packet
                packet = next(gen)
                # Transmit the packet
                yield self.flow.env.process(self.transmit(packet))
                # If we are in recovery mode
                if recover:
                    # Update the window size
                    self._update_window()
        # Once we have exhausted a packet generator
        except StopIteration:
            # Wait for acknowledgements in the last window
            yield self.flow.env.timeout(self.timeout)
            # While there remain dropped packets
            if self.timed_out:
                # Retransmit as many dropped packets as possible
                yield self.flow.env.process(self.recover())
            # If we exhausted the data packet generator
            if not recover:
                try:
                    # Kill the window control process
                    self._window_ctrl_proc.interrupt()
                except RuntimeError:
                    pass
                # Mark this TCP algorithm as finished if it isn't already
                if not self.finished.triggered:
                    self.finished.succeed()

    def recover(self):
        """Enter recovery mode.

        :return: None
        """
        # Halve the window size
        self.window = self.window / 2
        # Retransmit as many dropped packets as possible
        yield self.flow.env.process(self.burst(recover=True))

    def send(self):
        """Send all data packets.

        :return: None
        """
        # Start the window control process
        self._window_ctrl_proc = self.flow.env.process(self._window_control())
        # Yield a burst process
        yield self.flow.env.process(self.burst())


class Reno(TCP):
    """SimPy process implementing TCP Reno.

    :param flow: the flow to link this TCP instance to
    :type flow: :class:`Flow`
    :param int window: initial window size (packets)
    :param int timeout: packet acknowledgement timeout (ns)
    """

    def __init__(self, flow, window, timeout):
        super(Reno, self).__init__(flow, window, timeout)
        # flag indicating slow start phase
        self._slow_start = True
        # flag indicating congestion avoidance phase
        self._CA = False
        # flag indicating fast recovery phase
        self._fast_recovery = False
        # Slow start threshold
        self._threshold = 32
        # Next ID's expected by the destination host.  Last 4 are cached to
        # check for packet dropping (3 duplicate ACK's)
        self._next = deque(maxlen=4)
        # Waiting process for acknowledgements
        self._wait_proc = None

    def _wait(self):
        """Wait for acknowledgements; time out if none are recieved."""
        try:
            # Wait for acknowledgement(s)
            yield self.flow.env.timeout(self.timeout)
            # Retransmit dropped packets
            yield self.flow.env.process(self.burst(recover=True))
            # Set the slow start threshold to half the window size
            self._threshold = max(self.window / 2, 1)
            # Reset window size
            self.window = 1
            # Update monitored window size
            self.flow.env.update(
                "Window size,{},{},{}".format(self.flow.host.addr,
                                              self.flow.dest,
                                              self.flow.id),
                self.window)
            # Revert to slow start on timeout
            self._slow_start = True
            self._fast_recovery = False
            self._CA = False
        except simpy.Interrupt:
            # Reset the waiting process on interruption
            self._wait_proc = None

    def acknowledge(self, ack):
        """Process an acknowledgement packet.

        :param ack: the acknowledgement packet
        :type ack: :class:`resources.ACK`
        :return: None
        """
        # Stop the waiting process, if any
        if self._wait_proc:
            self._wait_proc.interrupt()
        # ID of the packet being acknowledged
        pid = ack.id
        # ID expected next by the destination host
        expected = ack.data
        # Push this into the deque (if the deque is full, this is
        # equivalent to self._next.popleft() before appending)
        self._next.append(expected)
        # Calculate round trip time
        rtt = self.flow.env.now - self.departure(pid)
        # Mark packet as acknowledged
        self.mark(pid)
        # If 2 * RTT is greater than the current timeout, update it
        self.timeout = min(max(self.timeout, 2 * rtt), self.max_time)
        #if in slow start phase
        if self._slow_start == True:
            self.window += 1
            if self.window >= self._threshold:
                self._slow_start = False
                self._CA = True
        # if in congestion avoidance phase
        elif self._CA == True:
            # If we have 3 duplicate ACK's, perform fast retransmission
            if self._next.count(self._next[0]) == self._next.maxlen:
                # Set slow start threshold to half of the window size
                self._threshold = self.window / 2
                # Set window size to the threshold + number of duplicates
                self.window = self._threshold + self._next.maxlen - 1
                # Enter fast recovery mode
                self._fast_recovery = True
                self._CA = False
                # Get the dropped packet
                dropped = self.packet(expected)
                # immediately retransmit the dropped packet
                yield self.flow.env.process(self.transmit(dropped))
                # wait for timeout
                self._wait_proc = self.flow.env.process(self._wait())
                yield self._wait_proc 
            else:
                #otherwise increment window size for successful ack
                self.window += 1 / self.window
        # if in fast recovery phase
        elif self._fast_recovery == True:
            # don't make any adjustments until there's a timeout, or
            # entire transmitted window is acknowledged
            if len(self.unacknowledged.items()) == 0:
                self._fast_recovery = False
                self._CA = True
        # Update monitored window size
        self.flow.env.update(
            "Window size,{},{},{}".format(self.flow.host.addr,
                                          self.flow.dest,
                                          self.flow.id),
            self.window)

    def burst(self, recover=False):
        """Inject packets into the network.

        If recover is True, then the packets are taken from the set of
        timed out packets.

        :param bool recover: flag for recovery mode
        :return: None
        """
        if recover:
            # Make a timed out packet generator
            gen = (p for p in self.timed_out[:])
        else:
            gen = self._gen
        try:
            # While we have room in the window
            while recover or len(self.unacknowledged) < int(self.window):
                # Get the next packet
                packet = next(gen)
                # Transmit the packet
                yield self.flow.env.process(self.transmit(packet))
            # Wait for acknowledgements, or time out
            self._wait_proc = self.flow.env.process(self._wait())
            yield self._wait_proc
        except StopIteration:
            if not recover:
                # While there are unacknowledged packets
                while self.unacknowledged:
                    # Wait for acknowledgements, or time out & recover
                    self._wait_proc = self.flow.env.process(self._wait())
                    yield self._wait_proc
                # Mark this TCP algorithm as finished, if it isn't already
                if not self._finished.triggered:
                    self._finished.succeed()

    def send(self):
        """Send all data packets.

        :return: None
        """
        while True and not self._finished.triggered:
            yield self.flow.env.process(self.burst())


@resources.reset
class Flow:
    """SimPy process representing a flow.

    Each flow process is connected to a source host, and generates as
    many packets as are necessary to send all of its data.  If data is
    None, a random, 12-bit number of packets are sent.  Every flow needs
    a TCP algorithm to be specified, from among the currently supported
    TCP algorithms.  At the moment, the only allowed specifiers are 
    \"FAST\", and \"Reno\".  See :class:`FAST` or :class:`Reno` for 
    details on what tcp_params should look like.

    :param simpy.Environment env: the simulation environment
    :param host: the source host of this flow
    :type host: :class:`Host`
    :param int dest: the address of the destination host
    :param int data: the total amount of data to transmit (bits)
    :param int delay: initial delay before sending any packets (ns)
    :param str tcp: TCP algorithm specifier
    :param list tcp_params: parameters for the TCP algorithm
    """

    allowed_tcp = {"FAST": FAST, "Reno": Reno}
    """A dict mapping TCP specifiers to :class:`TCP` subclasses."""

    def __init__(self, env, host, dest, data, delay, tcp, tcp_params):
        self._env = env
        # Flow host
        self._host = host
        # FLow destination
        self._dest = dest
        # Amount of data to transmit (bits)
        self._data = data
        # Time (ns) to wait before initial transmission
        self._delay = delay
        # Bits transmitted by flow
        self._transmitted = 0
        # Bits received by flow
        self._received = 0
        # List of roundtrip times for packets in given timestep
        self._times = list()
        # Register this flow with its host, and get its ID
        self._id = self._host.register(self)
        # Check for a valid TCP specifier
        try:
            # Initialize TCP object
            self._tcp = self.allowed_tcp[tcp](self, *tcp_params)
        except KeyError:
            raise ValueError("unsupported TCP algorithm \"{}\"".format(tcp))

        # Last mean round trip time
        self._mean_rtt = 0

        self.env.register(
            "Round trip times,{},{},{}".format(self._host.addr, self._dest,
                                               self._id),
            self._avg_rtt)
        self.env.register(
            "Flow rate,{},{},{}".format(self._host.addr, self._dest, self._id),
            lambda: self.reset("_transmitted"), True)

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
    def env(self):
        """The simulation environment.

        :return: SimPy environment
        :rtype: ``simpy.core.Environment``
        """
        return self._env

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

    def _avg_rtt(self):
        """Return averge round time trip."""
        try:
            avg = sum(self._times) / len(self._times)
            self._times = list()
            self._mean_rtt = avg
        except ZeroDivisionError:
            avg = self._mean_rtt
        return avg

    def _update_rtt(self, pid):
        """Update list of packet RTT's."""
        # Get departure time of packet
        depart = self._tcp.departure(pid)
        if depart is None:
            return
        # Calcualte roundtrip time of packet
        rtt = self.env.now - depart
        # append packet to list of RTT's
        self._times.append(rtt)

    def acknowledge(self, ack):
        """Receive an acknowledgement packet.

        Acknowledgement packets are passed to the underlying TCP algorithm
        to be handled.

        :param ack: acknowledgement packet
        :type ack: :class:`resources.ACK`
        :return: None
        """
        try:
            # Update round trip times
            self._update_rtt(ack.id)
            # update number of received bits
            self._received += resources.ACK.size
            # Send this acknowledgement to the TCP algorithm
            yield self.env.process(self._tcp.acknowledge(ack))
        except KeyError:
            logger.warning("Got two ACK\'s for packet {}".format(ack.id))

    def generate(self):
        """Generate packets from this flow.

        :return: None
        """
        yield self.env.timeout(self._delay)
        yield self.env.process(self._tcp.send())

    def generator(self):
        """Create a packet generator.

        If a data size is given, return a generator which yields
        ceil(data size / packet size) packets.  If data size is None,
        it yields a random, 12-bit number of packets.

        :return: packet generator
        :rtype: generator
        """
        n = 0
        if self._data == None:
            # Pick a random, 12-bit number of packets to send
            n = random.getrandbits(12)
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
        :return: None
        """
        #update transmitted number of bits
        self._transmitted += packet.size
        # Transmit the packet
        yield self.env.process(self._host.receive(packet))
        # Wait packet size / link capacity before terminating
        yield self.env.timeout(
            packet.size * 1e9 / self._host.transport.capacity)


@resources.reset
class Host:
    """SimPy process representing a host.

    Each host process has an underlying HostResource which handles
    (de)queuing packets.  This process handles interactions between the
    host resource, any active flows, and up to one outbound link.

    :param simpy.Environment env: the simulation environment
    :param int addr: the address of this host
    """

    def __init__(self, env, addr):
        # Initialize host resource
        self._res = resources.HostResource(env, addr)
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

        self.env.register("Host rate,{}".format(self._addr),
                          lambda: self.reset("_transmitted"), True)

    @property
    def addr(self):
        """The address of this host.

        :return: host address
        :rtype: int
        """
        return self._addr

    @property
    def env(self):
        """The simulation environment.

        :return: SimPy environment
        :rtype: ``simpy.core.Environment``
        """
        return self._res.env

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
        if packet.dest == self._addr:
            # update bits received by host
            self._received += packet.size

        # Queue new packet for transmission, and dequeue a packet. The
        # HostResource.receive event returns an outbound ACK if the
        # dequeued packet was an inbound data packet
        packet = yield self._res.receive(packet)

        # Transmit the dequeued packet
        yield self.env.process(self.transmit(packet))

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
            
            # update bits transmitted by host
            self._transmitted += packet.size
            # Transmit an outbound packet
            yield self.env.process(self._transport.send(packet))
        else:
            # Send an inbound ACK to its destination flow
            yield self.env.process(
                self._flows[packet.flow].acknowledge(packet))


@resources.reset
class Link:
    """SimPy process representing a link.

    Each link process has an underlying LinkResource which handles
    (de)queuing packets.  This process handles interactions between the
    link resource, and the processes it may connect.

    :param simpy.Environment env: the simulation environment
    :param int capacity: the link rate (bps)
    :param int size: the link buffer size (bits)
    :param int delay: the link delay in (ns)
    :param int lid: link id
    """

    def __init__(self, env, capacity, size, delay, lid):
        # Initialize link buffers
        self._res = resources.LinkBuffer(env, size, lid)
        # Link capacity (bps)
        self._capacity = capacity
        # Link delay (ns)
        self._delay = delay

        # Endpoints for each direction
        self._endpoints = [None, None]
        # "Download" handler
        self._down = Transport(self, resources.DOWN)
        # "Upload" handler
        self._up = Transport(self, resources.UP)
        # Total number of bits transmitted by link
        self._transmitted = [0, 0]
        # Buffer flushing processes
        self._flush_proc = (self.env.process(self._flush(resources.DOWN)),
                            self.env.process(self._flush(resources.UP)))
        # Time to wait between buffer pops if the buffer was empty
        self._flush_wait = 100000 # .1 ms

        self.env.register(
            "Link rate,{},{}".format(self._res.id, resources.DOWN),
            lambda: self.reset("_transmitted", resources.DOWN), True)
        self.env.register(
            "Link rate,{},{}".format(self._res.id, resources.UP),
            lambda: self.reset("_transmitted", resources.UP), True)

    @property
    def capacity(self):
        """The maximum bitrate of the link in bps.

        :return: link rate
        :rtype: int
        """
        return self._capacity

    @property
    def delay(self):
        """The link delay in nanoseconds.

        :return: link delay
        :rtype: int
        """
        return self._delay

    @property
    def endpoints(self):
        """A list of connected endpoints (up to 1 per direction)

        :return: connected endpoints
        :rtype: list
        """
        return self._endpoints

    @property
    def env(self):
        """The simulation environment.

        :return: SimPy environment
        :rtype: ``simpy.core.Environment``
        """
        return self._res.env

    @property
    def id(self):
        """Link id.

        :return: id
        :rtype: int
        """
        return self._res.id

    def _flush(self, direction):
        """Flush packets from the buffer at the link rate."""
        while True:
            # Dequeue a packet from the correct buffer
            packet = self._res.dequeue(direction)
            # If we got a packet
            if packet is not None:
                # Limit transmission speed to link bitrate
                yield self.env.timeout(
                    packet.size * 1e9 / self._capacity)
                # Transmit the packet to the conected endpoint
                self.env.process(self._transmit(direction, packet))
            else:
                # Otherwise, wait 1 ms and try again
                yield self.env.timeout(self._flush_wait)

    def _transmit(self, direction, packet):
        """Transmit a packet across the link in a given direction.

        This generating function yields a transmission process for each
        packet given

        :param int direction: the direction to transport the packet
        :param packet: a packet to transmit
        :type packet: :class:`resources.Packet`
        :return: None
        """
        # Transmit packet after waiting, as if sending the packet
        # across a physical link
        yield simpy.util.start_delayed(self.env,
            self._endpoints[direction].receive(packet), self._delay)
        # Update total bits transmitted by link
        self._transmitted[direction] += packet.size

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

        Total cost is calculated as propagation delay / log(capacity /
        data packet size) + cumulative buffer occupancy.

        :param int direction: link direction
        :return: directional link cost
        :rtype: float

        """
        return self._delay / math.log(self._capacity / resources.Packet.size) \
            + self._res.buffered(direction)

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
        # Enqueue the new packet, and get the dropped status
        dropped = yield self._res.enqueue(direction, packet)
        if not dropped:
            # update total buffer fill
            self._res.update_buffered(direction, self.env.now)


class Router:
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
        self._res = resources.PacketQueue(env, addr)
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
        self._last_arrival = float("inf")
        # Flag indicating if router has converged
        self._converged = False
        # timeout duration used to determine routing table convergence
        self._timeout = 50000000 # 50 ms
        # routing update period (only relevant for routers connected to hosts)
        self._bf_period = 5000000000 # 5 s

    @property
    def addr(self):
        """The address of this router.

        :return: router address
        :rtype: int
        """
        return self._addr

    @property
    def env(self):
        """The simulation environment.

        :return: SimPy environment
        :rtype: ``simpy.core.Environment``
        """
        return self._res.env

    @property
    def routers(self):
        """Transport handlers connected to routers.

        :return: a list of outbound transport handlers
        :rtype: [:class:`Transport`]
        """
        return [t for t in self._links 
                if Host not in map(type, t.link.endpoints)]

    def _broadcast_packet(self, host, cost, path, rec_port):
        """Broadcast a routing packet."""
        # Update the routing path
        new_path = path + [self._addr]
        # create a new routing packet for each outbound link that's
        # not connected a host
        for t in self.routers:
            if t == rec_port:
                continue
            # Create a new routing packet with updated rounting information
            r = resources.Routing((host, cost + t.cost, new_path, t.reverse()))
            # send the newly created packet
            yield self.env.process(self.transmit(r, t))

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
            # extract data from payload
            host, cost, path, rec_port = packet.data
            # update last arrival time of packet
            self._last_arrival = self.env.now
            # if packet has already gone through router, ignore it
            if self._addr in path:
                return     
            # update new routing table if host isn't in table
            if not (host in self._update_table.keys()):
                self._update_table[host] = (rec_port, cost)
                yield self.env.process(
                    self._broadcast_packet(host, cost, path, rec_port))
            # update new routing table if there's a more efficient path
            if self._update_table[host][1] > cost:
                self._update_table[host] = (rec_port, cost)
                yield self.env.process(
                    self._broadcast_packet(host, cost, path, rec_port))
            # after receiving a routing packet, begin update timeout
            yield self.env.timeout(self._timeout)
            # check to see if router has reached threshold (for the first time)
            if self.env.now >= self._last_arrival + self._timeout and \
                not self._converged:
                # set flag
                self._converged = True
                # For each neighboring router
                for t in self.routers:
                    # Send a Finish packet pointing to this router to inform
                    # them that this router has converged
                    yield self.env.process(
                        self.transmit(resources.Finish(t.reverse()), t))

        # check for one degree of convergence (this router and neighbors)
        if self._converged and all(self._finish_table.values()):
            # If there is an updated routing table
            for key, value in self._update_table.items():
                self._routing_table[key] = value
            # and reset all variables used for tracking convergence
            self._last_arrvial = float("inf")
            self._converged = False
            self._finish_table = dict()
            self._update_table = dict()
            # Reset buffer occupancy for new routing table cost calculations
            for l in self._links:
                l.link._res._occupancy = [0, 0]

    def route(self, address):
        """Return the correct transport handler for the given address.

        Raises a KeyError if the given address is not in the routing table.

        :param int address: the destination address
        :return: the transport handler selected by the routing policy
        :rtype: :class:`Transport`
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
                        host = next(filter(lambda e: type(e) == Host,
                                           transport.link.endpoints))
                        # update routing table for this router
                        self._update_table[host.addr] = (transport, 
                                                             transport.cost)
                        # For each connected router
                        for t in self.routers:
                            # Create routing packet for the connected host
                            packet = resources.Routing(
                                (host.addr, transport.cost + t.cost, 
                                 [self._addr], t.reverse()))
                            # Send the routing packet
                            yield self.env.process(
                                self.transmit(packet, t))
                    else:
                        # initialize _finish_table for neighboring routers
                        self._finish_table[transport] = False
                yield self.env.timeout(self._bf_period)
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

        # Push another packet through the queue
        packet = yield self._res.receive(packet)
        # first determine what type of packet it is, then process it 
        if isinstance(packet, resources.Routing):
            yield self.env.process(self._handle_routing_packet(packet))
        # handle data packet
        else:
            # look up outbound link using the routing table
            transport = self.route(packet.dest)
            # tranmit packet
            yield self.env.process(self.transmit(packet, transport))

    def transmit(self, packet, transport):
        """Transmit an outbound packet.        
       
        :param packet: the outbound packet     
        :type packet: :class:`resources.Packet`  
        :param transport: the outbound transport handler
        :type transport: :class:`Transport`      
        :return: None      
        """              
        # Send the packet      
        yield self.env.process(transport.send(packet))
        # Wait packet size / link rate before terminating
        yield self.env.timeout(packet.size * 1e9 / transport.capacity)


class Transport:
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
        """Capacity of the underlying link in bps.

        :return: link rate
        :rtype: int
        """
        return self._link.capacity

    @property
    def cost(self):
        """Directional cost of the underlying link.

        :return: link cost
        :rtype: float
        """
        return self._link.cost(self._direction)

    @property
    def direction(self):
        """The direction of this transport handler.

        :return: link direction
        :rtype: int
        """
        return self._direction

    @property
    def link(self):
        """The link that owns this transport handler.

        :return: underlying link
        :rtype: :class:`Link`
        """
        return self._link

    def reverse(self):
        """Return a transport handler in the opposite direction.

        :return: transport handler
        :rtype: :class:`Transport`
        """
        return Transport(self._link, 1 - self._direction)

    def send(self, packet):
        """Send a packet across the link.

        :param packet: the packet to send
        :type packet: :class:`resources.Packet`
        :return: :func:`Link.receive` (generating method)
        :rtype: generator
        """
        return self._link.receive(self._direction, packet)

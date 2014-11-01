import math
import queue
import random
import simpy

import resources

# TODO: use processes to control interaction between network resources

class Host(object):
    """SimPy process representing a host.

    Each host process has an underlying HostResource which handles
    (de)queuing packets.  This process handles interactions between the
    host resource, any active flows, and up to one outbound link.
    """

    def __init__(self, env, addr):
        self.res = resources.HostResource(env, addr)
        # Host address
        self._addr = addr
        # Active flows
        self._flows = list()
        # Outbound link
        self._transport = None

    @property
    def addr(self):
        """The address of this host."""
        return self._addr

    @property
    def flows(self):
        """A list of flows active on this host."""
        return self._flows

    def connect(self, transport):
        """Connect a new (link) transport handler to this host."""
        self._transport = transport

    def disconnect(self, transport):
        """Disconnect an existing transport handler.

        If the given transport handler is not connected to this host,
        do nothing.
        """
        if transport == self._transport:
            self._transport = None

    def register(self, flow):
        """Register a new flow on this host, and return the flow ID."""
        self._flows.append(flow)
        return len(self._flows) - 1

    def receive(self, packet):
        """Receive a packet.

        Packets may be inbound or outbound, data or acknowledgement
        packets.  Inbound data packets automatically generate an
        outbound acknowledgement.
        """
        # Queue new packet for transmission, and dequeue a packet. The
        # HostResource.receive event returns an outbound ACK if the
        # dequeued packet was an inbound data packet
        packet = yield self.res.receive(packet)
        # Transmit the dequeued packet
        yield self.res._env.process(self._transmit(packet))

    def _transmit(self, packet):
        """Transmit a packet."""
        if packet.dest != self._addr:
            # Transmit an outbound packet
            yield self.res._env.process(self._transport(packet))
        else:
            # Send an inbound ACK to its destination flow
            yield self.res._env.process(
                self._flows[packet.flow].acknowledge(packet))


class Link(object):
    """SimPy process representing a link.

    Each link process has an underlying LinkResource which handles
    (de)queuing packets.  This process handles interactions between the
    link resource, and the processes it may connect.
    """

    def __init__(self, env, capacity, delay, buf_size):
        self.res = resources.LinkResource(env, buf_size)
        # Link rate (bps)
        self._capacity = capacity
        # Link traffic (bps)
        self._traffic = [0, 0]
        # Link delay
        self._delay = delay

        # Endpoints for each direction
        self._endpoints = [None, None]
        # 'Upload' handler
        self._up = lambda p: self.receive(1, p)
        # 'Download' handler
        self._down = lambda p: self.receive(0, p)

    @property
    def capacity(self):
        """The maximum bitrate of the link in Bps."""
        return self._capacity

    @property
    def delay(self):
        """The link delay in seconds."""
        return self._delay

    @property
    def endpoints(self):
        """A list of connected endpoints (=< 1 per direction)"""
        return self._endpoints

    def connect(self, proc0, proc1):
        """Connect two network components via this link."""
        self._endpoints = [proc0, proc1]
        self._endpoints[0].connect(self._up)
        self._endpoints[1].connect(self._down)

    def disconnect(self):
        """Disconnect a link from its two endpoints."""
        self._endpoints[0].disconnect(self._up)
        self._endpoints[1].disconnect(self._down)
        self._endpoints = [None, None]

    def receive(self, direction, packet):
        """Receive a packet to transmit in a given direction."""
        # Queue the new packet for transmission, and dequeue a packet
        packet = yield self.res.transport(direction, packet)
        # Transmit the dequeued packet
        yield self.res._env.process(self._transmit(direction, packet))

    def _transmit(self, direction, packet):
        """Transmit a packet across the link in a given direction"""
        # If the link isn't busy
        if packet.size + self._traffic[direction] <= self._capacity:
            # Increment directional traffic
            self._traffic[direction] += packet.size
            # # Wait as if transmitting packet across a physical link
            # yield self.res._env.timeout(self._delay)
            # # Transmit the packet
            # yield self.res._env.process(
            #     self._endpoints[direction].receive(packet))
            # Transmit packet after waiting, as if sending the packet
            # across a physical link
            yield simpy.util.start_delayed(self.res._env,
                self._endpoints[direction].receive(packet), self._delay)
            # Decrement directional traffic
            self._traffic[direction] -= packet.size

    def static_cost(self):
        """Calculate the static cost of this link.

        Cost is inversely proportional to link capacity, and  directly
        proportional to delay.

        """
        try:
            return self._delay / self._capacity 
        # If the link is down (capacity == 0 bps), return infinite cost
        except ZeroDivisionError:
            return float("inf")

    def dynamic_cost(self, direction):
        """Calculate the dynamic cost of a direction on this link.

        Dynamic cost is directly proportional to link traffic and buffer
        fill.
        """
        return self._traffic[direction] * self.res.fill(direction)

    def cost(self, direction):
        """Return the total cost of a direction on this link.

        Total cost is simply calculated as static cost + dynamic cost
        """
        return self.static_cost() + self.dynamic_cost(direction)


class Flow(object):
    """SimPy process representing a flow.

    Each flow process is connected to a source host, and generates as
    many packets as are necessary to send all of its data.  If data is
    None, a random, 8-bit number of packets are sent.
    """

    def __init__(self, env, host, dest, window, timeout, data):
        self._env = env
        # Flow host
        self._host = host
        # FLow destination
        self._dest = dest
        # Window size for this flow
        self._window = window
        # Time (in simulation time) to wait for an acknowledgement before
        # retransmitting a packet
        self._time = timeout
        # Amount of data to transmit (bits)
        self._data = data

        # Initialize packet generator
        self._packets = self._generator()
        # Initialize list of sent packets
        self._sent = list()
        # Register this flow with its host, and get its ID
        self._id = self._host.register(self)
        # Wait to receive acknowledgement
        self._wait_for_ack = self._env.event()

    def _generator(self):
        """Create a packet generator.

        If a data size is given, return a generator which yields
        ceil(data size / packet size) packets.  If data size is None,
        yield a random, 8-bit number of packets.

        """
        n = 0
        if self._data == None:
            # Pick a random 8-bit number of packets to send
            n = random.getrandbits(8)
        else:
            # Calculate how many packets are needed to send self._data bits
            n = math.ceil(self._data / resources.Packet.size)

        # Create packet generator
        g = (resources.Packet(self._host.addr, self._dest, self._id, i, 1)
             for i in range(n))

        return g

    def generate(self):
        while True:
            try:
                # Send as many packets as fit in our window
                for i in range(self._window):
                    # Generate the next packet to send
                    packet = next(self._packets)
                    # Append the packet to our sent list
                    self._sent.append(packet)
                    # Send the packet through the connected host
                    yield self._env.process(self._host.receive(packet))
                # Passivate packet generation
                yield self._wait_for_ack
            except StopIteration:
                # Stop trying to send packets once we run out (last window
                # is exhausted)
                break
        # Wait for the packets in the last window
        yield self._wait_for_ack

    def acknowledge(self, ack):
        # Acknowledge the packet whose ACK was received
        self._packets[ack.id] = None
        # Wait for other acknowledgements
        yield self._env.timeout(self._time)
        # Determine which packets in this window have not been acknowledged
        targets = filter(lambda p: p is not None, 
                         self._packets[-self._window]))
        # Resend dropped packets
        for packet in targets:
            yield self._env.process(self._host.receive(packet))

        # TODO: congestion control algorithm should (potentially) modify
        #       self._window here

        # Continue sending more packets
        self._wait_for_ack.succeed()
        self._wait_for_ack = self._env.event()



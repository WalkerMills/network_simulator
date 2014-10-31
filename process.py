import queue
import resources
import simpy

# TODO: use processes to control interaction between network resources

class Host:
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
                self._flows[packet.flow].receive(packet))


class Link:
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
            # Wait as if transmitting packet across a physical link
            yield self.res._env.timeout(self._delay)
            # Transmit the packet
            yield self.res._env.process(
                self._endpoints[direction].receive(packet))
            # Decrement directional traffic
            self._traffic[direction] -= packet.size

    # TODO: implement dynamic cost method
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

    def cost(self, direction):
        """Calculate the dynamic cost of a direction on this link.

        Dynamic cost is directly proportional to link delay and
        utilization.
        """
        return self._traffic[direction] * self.static_cost()


class Flow(object):
    """Simulator object representing a flow."""

    def __init__(self, id, env, src, dest, host, data=None):
        self.env = env

        # flow id
        self.id = id
        # Source address
        self.src = src
        # Destination address
        self.dest = dest
        # Flow start time
        self.start = env.now
        # Owner host
        self.host = host
        # Queue to hold packets to transmit to use for flow control
        self.packets = queue.Queue()

        # TODO: check data for a valid value

        # Flag to determine finite data output; None specifies infinite
        # random data generation
        self.data = None

    def generate(self):
        """Generate packets of size 1024 bytes to send."""

        # create a new packet with the same environment, id, src, and dest
        # make up some payload, and let it be "TESTING"
        packet = Packet(self.env, self.id, self.src, self.dest, "TESTING") # TODO
        
        while True:
            # After generating the packet, the Flow requests access to the 
            # host to send it.
            with self.host.host.request() as req:
                yield self.env.timeout(5)
                yield req
                
                # Forwards the packet
                yield self.env.process(self.host.transmit(packet))
                
                # Lets go of host after recieving acknowledgement
                yield self.env.process(self.host.receiveack(self.env))
            # TODO go back and add data collection.

    def stop_and_wait(self, env, timeout):
        """The simplest form of flow control of stop and wait"""
        pass

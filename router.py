import simpy


class ReceivePacket(simpy.events.Event):
    """Simulator event representing packet reception.

    This event takes a router as a parameter, and represents the addition
    of a packet to that router.  It also triggers a packet transmission
    for each packet received.  This event may also be used as a context
    manager.
    """

    def __init__(self, resource, packet):
        # Initialize event
        super(ReceivePacket, self).__init__(resource._env)
        self.resource = resource
        self.packet = packet
        self.proc = self.env.active_process

        print('Got packet at time {}'.format(self.env.now))
        # Add this event to the packet queue
        resource.packets.append(self)
        # Send a packet, since we have enqueued a new packet
        self.resource._trigger_send()

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        # If the event was interrupted, dequeue it
        if not self.triggered:
            self.resource.packets.remove(self)

    cancel = __exit__


class Router(object):
    """Simulator object representing a router.

    Router is implemented as a unidirectional resource.  It receives
    packets, but does not offer packet transmission requests.
    """
    # Queue of packet events to process
    PacketQueue = list

    def __init__(self, env):
        self._env = env
        self.packets = self.PacketQueue()
        # List of outbound links of the form (link cost, link)
        self.links = list()
        # Dictionary mapping outbound links to destination ranges
        self.table = dict()

        # Bind event constructors as methods
        simpy.core.BoundClass.bind_early(self)

    receive = simpy.core.BoundClass(ReceivePacket)

    def _send(self, packet):
        """Send an outbound packet."""
        # TODO: route packet, send it through a link
        pass

    def _trigger_send(self):
        """Trigger outbound packet transmission."""
        event = self.packets.pop(0)
        self._send(event.packet)
        event.succeed()

    def register_link(self, transport):
        """Register an outbound link (transport handler)."""
        self.links.append((transport.cost(), transport))

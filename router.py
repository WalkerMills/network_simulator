import packet
import simpy


class Router(object):
    """Simulator object representing a router.

    Router is implemented as a unidirectional resource.  It receives
    packets, but does not accept packet transmission requests.  Every
    packet arrival triggers a packet transmission, at which time the
    next packet in the queue is popped, routed, and sent through the
    appropriate link.
    """

    def __init__(self, env, dynamic=False):
        self._env = env
        self._dynamic = dynamic
        # Queue of packet events to process
        self._packets = list()
        # List of outbound links of the form (link cost, link)
        self._links = list()
        # Dictionary mapping outbound links to destination ranges
        self._table = dict()

        # Bind event constructors as methods
        simpy.core.BoundClass.bind_early(self)

    receive = simpy.core.BoundClass(packet.ReceivePacket)

    @property
    def dynamic(self):
        return self._dynamic

    @dynamic.setter
    def dynamic(self, d):
        if d not in [True, False]:
            raise ValueError("Invalid dynamic flag value.")

        self._dynamic = d

    def route(self, dest, dynamic=False):
        in_range = lambda l: dest in self._table[l[1]]
        # TODO: dynamic routing updates link costs before choosing
        out = min(filter(in_range, self._links))

        return out

    def _transmit(self, packet):
        """Transmit an outbound packet."""
        # TODO: route packet & transmit it through a link
        pass

    def _trigger_transmit(self):
        """Trigger outbound packet transmission."""
        event = self._packets.pop(0)
        self._transmit(event.packet)
        event.succeed()

    def register_link(self, transport):
        """Register an outbound link (transport handler)."""
        self._links.append((transport.cost(), transport))
        # TODO: add link endpoints to self._table

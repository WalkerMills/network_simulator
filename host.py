import packet
import simpy


class Host(object):
    """Simulator object representing a host.
    
    Hosts are implemented as a unidirectional resource.  It receives
    packets, but does not accept packet transmission requests.  Every
    packet arrival triggers a packet transmission, at which time the
    next packet in the queue is popped.  Inbound data packets trigger
    an automatic acknowledgement, and inbound ACK packets are sent to
    the appropriate flow.  Outbound packets are transmitted via the
    outbound link, if it exists.

    """

    def __init__(self, env, addr):
        self._env = env
        # host address
        self._addr = addr
        # Queue to hold packets to transmit
        self._packets = list()
        # Outbound link
        self._link = None

        simpy.core.BoundClass.bind_early(self)

    @property
    def addr(self):
        return self._addr

    receive = simpy.core.BoundClass(packet.ReceivePacket)

    def _transmit(self, packet):
        """transmit an outbound packet."""
        if packet.dest == self.addr:
            # TODO: send ACK back to source flow
            pass
        else:
            # TODO: send outbound packet through self.link
            pass

    def _trigger_transmit(self):
        """Trigger outbound packet transmission."""
        event = self._packets.pop(0)

        # If a data packet has reached its destination, transmit an ACK
        if event.packet.dest == self.addr && event.packet.size == 1024:
            ack = packet.ACK(self.addr, event.packet.src)
            self._transmit(ack)
        # Otherwise, transmit the packet
        else:
            self._transmit(event.packet)
        event.succeed()

    def register_link(self, transport):
        """Register outbound link (transport handler)."""
        self._link = transport

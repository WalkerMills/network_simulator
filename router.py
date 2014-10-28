import simpy


class ReceivePacket(simpy.events.Event):

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

    PacketQueue = list

    def __init__(self, env):
        self._env = env
        self.packets = self.PacketQueue()
        self.links = list()

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

        self.links.append(transport)

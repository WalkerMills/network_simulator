import simpy


class RouterPut(simpy.resources.base.Put):
    """Add a packet to the router queue"""

    def __init__(self, resource, packet):
        super(RouterPut, self).__init__(resource)
        self.packet = packet


class RouterGet(simpy.resources.base.Get):
    """Send a packet from the router queue"""
    pass


class Router(simpy.resources.base.BaseResource):
    """Simulator object representing a router."""

    def __init__(self, env):
        self.env = env
        self.packets = list()

        # TODO: cache cost for each outbound link
        self.links = list()

    put = simpy.core.BoundClass(RouterPut)

    get = simpy.core.BoundClass(RouterGet)

    def _do_put(self, event):
        """Put a packet into the router queue."""

        # Add the packet to the router's queue
        self.packets.append(event.packet)
        # Add a callback to send a packet
        event.callbacks.append(self._do_get)
        event.succeed()

    def _do_get(self, event):
        """Send a packet from the queue."""

        # TODO: each packet needs to be routed down a link
        event.succeed(self.packets.pop(0))

    def register_link(self, transport):
        # Register an outbound link (transport handler)
        self.links.append(transport)

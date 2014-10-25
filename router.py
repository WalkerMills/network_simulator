import simpy


class RouterPut(simpy.resources.base.Put):

    def __init__(self, resource, packet):
        super(RouterPut, self).__init__(resource)
        self.packet = packet


class Router(simpy.resources.base.BaseResource):
    """Simulator object representing a router."""

    def __init__(self, env):
        self.env = env
        self.packets = list()

    put = simpy.core.BoundClass(RouterPut)

    def _do_put(self, event):
        """Put a packet into the router queue."""

        self.packets.append(event.packet)
        event.succeed()

    def _do_get(self, event):
        """Flush the packet queue."""

        # TODO: each packet needs to be routed

        for packet in self.packets:
            event.succeed(self.packets.pop(0))

import simpy

class Host:
    """Simulator object representing a host"""

    def __init__(self, env, addr):
        self.env = env

        # host name. DNS translates this into an IP address
        self.name = name
        # host address
        self.addr = addr
        # Queue to hold packets to transmit
        self.packets = Queue()

    def transmit(self, env):
        """Transmit packets to their destination."""
        pass

    def receive(self, env):
        """Receive & forwards an inbound (acknowledgement) packet."""
        pass

    def acknowledge(self, env):
        """Transmit acknowledgement packet of size 64 bytes"""
        pass
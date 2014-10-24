import simpy

class Flow:
    """Simulator object representing a flow."""

    def __init__(self, env, src, dest, data=None):
        self.env = env

        # Source address
        self.src = src
        # Destination address
        self.dest = dest

        # TODO: check data for a valid value

        # Flag to determine finite data output; None specifies infinite
        # random data generation
        self.data = None

    def packets(self):
        """Generate packets to send."""
        pass

    def transmit(self, env):
        """Transmit packets to their destination."""
        pass

    def receive(self, env):
        """Receive an inbound (acknowledgement) packet."""
        pass
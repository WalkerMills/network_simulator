import simpy
import time

class Flow:
    """Simulator object representing a flow."""

    def __init__(self, id, env, src, dest, data=None):
        self.env = env

        # flow id
        self.id = id
        # Source address
        self.src = src
        # Destination address
        self.dest = dest
        # Flow start time
        self.start = time.perf_counter()

        # TODO: check data for a valid value

        # Flag to determine finite data output; None specifies infinite
        # random data generation
        self.data = None

    def generate(self):
        """Generate packets of size 1024 bytes to send."""
        pass

    def transmit(self, env):
        """Transmit packets to their destination."""
        pass

    def receive(self, env):
        """Receive an inbound (acknowledgement) packet."""
        pass

    def stop_and_wait(self, env, timeout):
        """The simplest form of flow control of stop and wait"""

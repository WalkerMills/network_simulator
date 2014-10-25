import simpy

class Host:
    """Simulator object representing a host"""

    def __init__(self, env, addr):
        self.env = env

        # host name. DNS translates this into an IP address
        self.name = name
        # host address
        self.addr = addr

    def acknowledge(self, env):
        """Transmit acknowledgement packet of size 64 bytes"""
        pass

    
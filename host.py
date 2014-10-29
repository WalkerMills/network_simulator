import simpy

class Host(object):
    """Simulator object representing a host
    A host can only send 1 packet at a timeout

    Flows have to request for the host. When they have access
    to one, they can start the sending process and wait for it 
    to finish and to receive the acknowledgement packet
    """

    def __init__(self, env, addr, sendtime, receivetime):
        self.env = env
        # host name. DNS translates this into an IP address
        self.name = name
        # host address
        self.addr = addr
        # Queue to hold packets to transmit
        self.packets = Queue()
        # Create a shared resource to limit flows' access
        self.host = simpy.Resource(env, 1)
        # This is the time it takes for the host to send a packet
        self.sendtime = sendtime
        # This is the time it takes for a host to receive a packet
        self.receivetime = receivetime

    def transmit(self, packet):
        """Transmit packets to their destination."""
        yield self.env.timeout(self.sendtime)

    def transmitack(self, env, packet, senderaddr):
        """Transmit acknowledgement packet of size 64 bytes"""
        yield self.env.timeout(self.sendtime)

    def receive(self, env, packet, senderaddr):
        """Receive inbound packet."""
        ack = Packet()
        yield self.env.timeout(self.receivetime)
        transmitack(self, env, ack, senderaddr)

    def receiveack(self, env):
        """ Receive an inbound acknowledgement packet"""
        while True:
            if (ack is received):
                break

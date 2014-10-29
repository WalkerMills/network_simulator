<<<<<<< HEAD
class Packet(object):
    """This class represents a packet in our simulation."""

    def __init__(self, src, dest):
        self._src = src
        self._dest = dest
        self._size = 1024

    @property
    def src(self):
        """Return the packet's source address."""

        return self._src

    @property
    def dest(self):
        """Return the packet's destination address."""

        return self._dest

    @property
    def size(self):
        """Return the packet size in bytes."""

        return self._size


class ACK(Packet):
    """This class represents an acknowedgement packet."""

    def __init__(self, src, dest):
        super(ACK, self).__init__(src, dest)
        self._size = 64
=======
import simpy

class Packet:
	"""Simulator object representing a packet"""

	def __init__(self, env, flow_id, src, dest, payload):
		self.env = env

		# The flow id of the flow that generated this packet
		self.flow_id = flow_id
		# Source address of packet
		self.src = src
		# Destination address of packet
		self.dest = dest
		# Data of the packet
		self.payload = payload

		# acknolwedgement packet is size 64 bytes
		# normal packet is size 1024 bytes

		# TODO: identification tag to reconstruct packet pieces
>>>>>>> flowhost

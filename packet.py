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
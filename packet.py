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

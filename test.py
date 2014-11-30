"""
.. module:: test
    :platform: Unix
    :synopsis: Test cases & monitoring for the network simulator.
"""

import enum
import heapq
import itertools
import simpy

import process

@enum.unique
class Case(enum.Enum):
    """An enumeration defining different test cases."""

    zero = 0
    """Test case 0."""
    one = 1
    """Test case 1."""
    two = 2
    """Test case 2."""

ZERO = Case.zero
"""Convenience alias for test case 0 (type)."""
ONE = Case.one
"""Convenience alias for test case 1 (type)."""
TWO = Case.two
"""Convenience alias for test case 2 (type)."""


class TestCase(object):
    """Factory class for producing test case adjacency lists.

    See ``adjacent`` for details.
    """

    adjacencies = {
        Case.zero: ([(('h0', 'h1'), (80000000, 512000, 10))], 
                    [(('h0', 'h1'), (160000000, 1000))]),
        Case.one: ([(('h0', 'r0'), (100000000, 512000, 10)),
                    (('r0', 'r1'), (80000000, 512000, 10)),
                    (('r0', 'r3'), (80000000, 512000, 10)),
                    (('r1', 'r2'), (80000000, 512000, 10)),
                    (('r3', 'r2'), (80000000, 512000, 10)),
                    (('r2', 'h1'), (100000000, 512000, 10))],
                   [(('h0', 'h1'), (160000000, 500))]),
        Case.two: ([(('r0', 'r1'), (80000000, 1024000, 10)),
                    (('r1', 'r2'), (80000000, 1024000, 10)),
                    (('r2', 'r3'), (80000000, 1024000, 10)),
                    (('h0', 'r0'), (100000000, 1024000, 10)),
                    (('h1', 'r0'), (100000000, 1024000, 10)),
                    (('h2', 'r2'), (100000000, 1024000, 10)),
                    (('h3', 'r3'), (100000000, 1024000, 10)),
                    (('h4', 'r1'), (100000000, 1024000, 10)),
                    (('h5', 'r3'), (100000000, 1024000, 10))],
                   [(('h0', 'h3'), (280000000, 500)),
                    (('h1', 'h4'), (120000000, 10000)),
                    (('h2', 'h5'), (240000000, 20000))])
    }
    """Edge & flow adjacency lists for each test case."""

    tcp_parameters = {
        Case.zero: {'FAST': [[1, 50, 384000]], 'Reno': [[1, 50]]}, 
        Case.one: {'FAST': [[1, 100, 384000]], 'Reno': [[1, 100]]},
        Case.two: {'FAST': itertools.repeat([1, 200, 768000], 3), 
                   'Reno': itertools.repeat([1, 200], 3)}
    }
    """TCP parameters for each test case."""

    @classmethod
    def adjacent(cls, case, tcp="FAST"):
        """Generate an adjanceny list for a given test case.

        For the given test case & TCP specifier, an adjacency list of
        the type taken by :class:`network.Network` is constructed from
        the adjacency lists in :attr:`adjacencies` and the TCP parameters
        in :attr:`tcp_parameters`.

        :param case: test case to build adjacency lists for
        :type case: :class:`Case`
        :param str tcp: TCP specifier (see :data:`process.Flow.allowed_tcp`)
        :return: Adjacency lists defining edges & flows
        :rtype: ([((str, str), (int, int, int))], 
                 [((str, str), ((int, int), (str, list)))])
        """
        # Check for valid parameters
        if not isinstance(case, Case):
            raise ValueError("invalid test case \'{}\'".format(case))
        if tcp not in process.Flow.allowed_tcp.keys():
            raise ValueError("invalid TCP specifier \'{}\'".format(tcp))

        # Initialize list of flows
        flows = list()
        for (tags, param), tcp_param in zip(cls.adjacencies[case][1], 
                                            cls.tcp_parameters[case][tcp]):
            # Add a flow with the proper TCP parameters
            flows.append((tags, (param, (tcp, tcp_param))))
        # Return adjacency lists of edges & initialized flows
        return cls.adjacencies[case][0], flows


class MonitoredEnvironment(simpy.Environment):
    """SimPy environment with monitoring.

    Processes may register identifiers, along with a getter (function)
    to be periodically called, and its return value recorded.  After the
    last event at a given time are processed, all monitored values are
    updated.  However, monitored values are only updated after an event
    occurs, i.e., if no events occur at a given time, no update is
    performed.

    :param int initial_time: simulation time to start at
    """

    def __init__(self, initial_time=0):
        super(MonitoredEnvironment, self).__init__(initial_time)

        # Dictionary mapping identifier -> getter
        self._getters = dict()
        # Dictionary mapping identifier -> [(time, monitored value)]
        self._monitored = dict()

    def monitored(self):
        """The timestamped values of all monitored attributes.

        :return: monitored attribute dict
        :rtype: {str: [(int, object)]}
        """
        return self._monitored

    def values(self, name):
        """The values for the given identifier.

        :param str name: the identifier to retrieve values for
        :return: timestamped values list
        :rtype: [(int, object)]
        """
        return self._monitored[name]

    def _update(self):
        """Update all monitored values."""
        # For each identifier
        for name, getter in self._getters.keys():
            # Append a new timestamped value to the list 
            self._monitored[name].append((self._now, getter()))

    def register(self, name, getter):
        """Register a new identifier.

        Raise a KeyError if the given identifier already exists.

        :param str name: the identifier
        :param function getter: a function to update the monitored value with
        :return: None
        """
        # Don't accept duplicate identifiers
        if name in self._getters.keys():
            raise KeyError("already monitoring {}".format(name))

        # Add this identifier to the getter & values dictionaries
        self._getters[name] = getter
        self._monitored[name] = []

    def step(self):
        """Process the next event, and update the monitored values.

        Raise an :exc:`simpy.core.EmptySchedule` if no further events
        are available.

        :return: None
        """
        try:
            self._now, _, _, event = heapq.heappop(self._queue)
        except IndexError:
            raise simpy.core.EmptySchedule()

        # Process callbacks of the event.
        for callback in event.callbacks:
            callback(event)
        event.callbacks = None

        # If the next event is in the future, or nonexistent 
        if self.peek() > self._now:
            # Update the monitored values
            self._update()

        if not event.ok and not hasattr(event, "defused"):
            # The event has failed, check if it is defused.
            # Raise the value if not.
            raise event._value

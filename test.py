"""
.. module:: test
    :platform: Unix
    :synopsis: Test cases & monitoring for the network simulator.
"""

import heapq
import simpy

ZERO = ([(('h0', 'h1'), (80000000, 512000, 10))], 
        [(('h0', 'h1'), ((160000000, 1000), ('FAST', [1, 50, 384000])))])
"""Test case 0."""

ONE = ([(('h0', 'r0'), (100000000, 512000, 10)),
        (('r0', 'r1'), (80000000, 512000, 10)),
        (('r0', 'r3'), (80000000, 512000, 10)),
        (('r1', 'r2'), (80000000, 512000, 10)),
        (('r3', 'r2'), (80000000, 512000, 10)),
        (('r2', 'h1'), (100000000, 512000, 10))],
       [(('h0', 'h1'), ((160000000, 500), ('FAST', [1, 100, 384000])))])
"""Test case 1."""
        
TWO = ([(('r0', 'r1'), (80000000, 1024000, 10)),
        (('r1', 'r2'), (80000000, 1024000, 10)),
        (('r2', 'r3'), (80000000, 1024000, 10)),
        (('h0', 'r0'), (100000000, 1024000, 10)),
        (('h1', 'r0'), (100000000, 1024000, 10)),
        (('h2', 'r2'), (100000000, 1024000, 10)),
        (('h3', 'r3'), (100000000, 1024000, 10)),
        (('h4', 'r1'), (100000000, 1024000, 10)),
        (('h5', 'r3'), (100000000, 1024000, 10))],
       [(('h0', 'h3'), ((280000000, 500), ('FAST', [1, 200, 768000]))),
        (('h1', 'h4'), ((120000000, 10000), ('FAST', [1, 200, 768000]))),
        (('h2', 'h5'), ((240000000, 20000), ('FAST', [1, 200, 768000])))])
"""Test case 2."""


class MonitoredEnvironment(simpy.Environment):

    def __init__(self, initial_time=0):
        super(MonitoredEnvironment, self).__init__(initial_time)

        # Dictionary mapping identifier -> getter
        self._getters = dict()
        # Dictionary mapping identifier -> [(time, monitored value)]
        self._monitored = dict()

    @property
    def monitored(self):
        """The timestamped values of all monitored attributes.

        :return: monitored attribute dict
        :rtype: {str: [(int, object)]}
        """
        return self._monitored

    def monitored(self, name):
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

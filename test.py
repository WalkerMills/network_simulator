"""
.. module:: test
    :platform: Unix
    :synopsis: Test cases & monitoring for the network simulator.
"""

import enum
import heapq
import itertools
import numpy as np
import simpy

from matplotlib import pyplot as plt

import network
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
        Case.zero: ([(('h0', 'h1'), (80000000, 512000, 10000000))], 
                    [(('h0', 'h1'), (160000000, 1000000000))]),
        Case.one: ([(('h0', 'r0'), (100000000, 512000, 10000000)),
                    (('r0', 'r1'), (80000000, 512000, 10000000)),
                    (('r0', 'r3'), (80000000, 512000, 10000000)),
                    (('r1', 'r2'), (80000000, 512000, 10000000)),
                    (('r3', 'r2'), (80000000, 512000, 10000000)),
                    (('r2', 'h1'), (100000000, 512000, 10000000))],
                   [(('h0', 'h1'), (160000000, 500000000))]),
        Case.two: ([(('r0', 'r1'), (80000000, 1024000, 10000000)),
                    (('r1', 'r2'), (80000000, 1024000, 10000000)),
                    (('r2', 'r3'), (80000000, 1024000, 10000000)),
                    (('h0', 'r0'), (100000000, 1024000, 10000000)),
                    (('h1', 'r0'), (100000000, 1024000, 10000000)),
                    (('h2', 'r2'), (100000000, 1024000, 10000000)),
                    (('h3', 'r3'), (100000000, 1024000, 10000000)),
                    (('h4', 'r1'), (100000000, 1024000, 10000000)),
                    (('h5', 'r3'), (100000000, 1024000, 10000000))],
                   [(('h0', 'h3'), (280000000, 500000000)),
                    (('h1', 'h4'), (120000000, 10000000000)),
                    (('h2', 'h5'), (240000000, 20000000000))])
    }
    """Edge & flow adjacency lists for each test case."""

    tcp_parameters = {
        Case.zero: {'FAST': [[1, 50000000, 384000]], 'Reno': [[1, 50000000]]}, 
        Case.one: {'FAST': [[1, 120000000, 384000]], 'Reno': [[1, 120000000]]},
        Case.two: {'FAST': itertools.repeat([1, 200000000, 768000], 3), 
                   'Reno': itertools.repeat([1, 200000000], 3)}
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

def run(adjacent, tcp="FAST", until=None):
    """Runs the simulation with the given parameters. And displays
    graphs for monitored variables

    :param adjacent: adjacency lists of links & flows defining a network
    :type adjacent: ([((str, str), (int, int, int))], 
        [((str, str), ((int, int), (str, list)))]), or :class:`test.Case`
    :param str tcp: TCP specifier. Used iff adjacent is a :class:`test.Case`
    :param until_: time or event to run the simulation until
    :type until_: int or ``simpy.events.Event``
    """
    sorted_data = dict()

    n = network.Network(adjacent, tcp)
    data = n.simulate(until)

    for key, value in data.items():
        separated = key.split(',')
        title = separated[0]
        if title not in sorted_data.keys():
            sorted_data[title] = list()
        # creates new dictionary of tuples which contain 
        # data points w/ unique host-flow identifier
        sorted_data[title].append((', '.join(separated[1:]), value))

    graph_data(sorted_data)
    return sorted_data

def graph_data(sorted_data):
    graph_args = {"Flow received": ["flow", "Mbps", True, 1000000],
                  "Flow transmitted": ["flow", "Mbps", True, 1000000],
                  "Round trip times": ["flow", "ms", False, 1000000],
                  "Host transmitted": ["host", "Mbps", True, 1000000],
                  "Host received": ["host", "Mbps", True, 1000000],
                  "Link fill": ["link", "packets", False, 1],
                  "Dropped packets": ["link", "packets", False, 1],
                  "Link transmitted": ["link", "Mbps", True, 1000000],
                  "Window size": ["flow", "packets", False, 1],
                  "Queueing delay": ["flow", "ns", False, 1]}

    for key, value in sorted_data.items():
        _graph(key, value, *graph_args[key])

def _graph(title, data, legend, y_label, derive=False, scaling=1):
    """
    """
    
    fig = plt.figure()
    fig.suptitle(title)
    plt.xlabel("simulation time (ms)")
    plt.ylabel(y_label)

    for dataset in data:
        arr = np.asarray(dataset[1])
        x, y = np.transpose(arr)
        y = [float(value) / float(scaling) for value in y]

        if derive:
            y = np.gradient(y)
        plt.plot(x, y, label="{} {}".format(legend, dataset[0]))

    # may need to insert legend here
    plt.legend()
    plt.show()
    plt.close('all')

    



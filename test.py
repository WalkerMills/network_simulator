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
        Case.one: {'FAST': [[1, 120, 384000]], 'Reno': [[1, 120]]},
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

    for key, value in sorted_data.items():
        if key == "Flow received":
            graph(value, key, "Mbps", True, 1000000)

        elif title == "Flow transmitted":
            graph(value, key, "Mbps", True, 1000000)

        elif title == "Flow rtt":
            graph(value, key, "ms")

        elif title == "Host transmitted":
            graph(value, key, "Mbps", True, 1000000)

        elif title == "Host received":
            graph(value, key, "ms", True, 1000000)

        elif title == "Link fill":
            graph(value, key, "packets")

        elif title == "Link dropped":
            graph(value, key, "packets")

        elif title == "Link transmitted":
            graph(value, key, "Mbps", True, 1000000)

    return data

def graph(data, title, y_label, derive=False, scaling=1):
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
            d = np.empty(len(y))
            # calculate derivative
            for i in range(len(y) - 1):
                d[i] = float(y[i+1] - y[i]) / float(x[i+1] - x[i])
            y = d

        plt.plot(x, y, label="flow{}".format(dataset[0]))

    # may need to insert legend here
    plt.show()

    



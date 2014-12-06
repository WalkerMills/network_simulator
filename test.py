"""
.. module:: test
    :platform: Unix
    :synopsis: Test cases & monitoring for the network simulator.
"""

import enum
import heapq
import logging
import numpy as np
import simpy

from matplotlib import pyplot as plt

import network
import process

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        Case.zero: ([(('h0', 'h1'), (10000000, 512000, 10000000))], 
                    [(('h0', 'h1'), (160000000, 1000000000))]),
        Case.one: ([(('h0', 'r0'), (12500000, 512000, 10000000)),
                    (('r0', 'r1'), (10000000, 512000, 10000000)),
                    (('r0', 'r3'), (10000000, 512000, 10000000)),
                    (('r1', 'r2'), (10000000, 512000, 10000000)),
                    (('r3', 'r2'), (10000000, 512000, 10000000)),
                    (('r2', 'h1'), (12500000, 512000, 10000000))],
                   [(('h0', 'h1'), (160000000, 500000000))]),
        Case.two: ([(('r0', 'r1'), (10000000, 1024000, 10000000)),
                    (('r1', 'r2'), (10000000, 1024000, 10000000)),
                    (('r2', 'r3'), (10000000, 1024000, 10000000)),
                    (('h0', 'r0'), (12500000, 1024000, 10000000)),
                    (('h1', 'r0'), (12500000, 1024000, 10000000)),
                    (('h2', 'r2'), (12500000, 1024000, 10000000)),
                    (('h3', 'r3'), (12500000, 1024000, 10000000)),
                    (('h4', 'r1'), (12500000, 1024000, 10000000)),
                    (('h5', 'r3'), (12500000, 1024000, 10000000))],
                   [(('h0', 'h3'), (280000000, 500000000)),
                    (('h1', 'h4'), (120000000, 10000000000)),
                    (('h2', 'h5'), (240000000, 20000000000))])
    }
    """Edge & flow adjacency lists for each test case."""

    tcp_parameters = {
        Case.zero: {'FAST': [[1, 50000000, 32]], 'Reno': [[1, 50000000]]}, 
        Case.one: {'FAST': [[1, 120000000, 20]], 'Reno': [[1, 100000000]]},
        Case.two: {'FAST': [[1, 150000000, 20], 
                            [1, 80000000, 20],
                            [1, 80000000, 20]],
                   'Reno': [[1, 150000000], 
                            [1, 80000000],
                            [1, 80000000]]}
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


class Graph(object):

    title_args = {"Flow received": ["flow", "Mbps", 1000],
                  "Flow transmitted": ["flow", "Mbps", 1000],
                  "Round trip times": ["flow", "ms", 1e-6],
                  "Host transmitted": ["host", "Mbps", 1000],
                  "Host received": ["host", "Mbps", 1000],
                  "Link fill": ["link", "packets"],
                  "Dropped packets": ["link", "packets"],
                  "Link transmitted": ["link", "Mbps", 1000],
                  "Window size": ["flow", "packets"],
                  "Queuing delay": ["flow", "ms", 1e-6]}
    """Maps graph title to :meth:`graph` parameters."""

    def __init__(self, adjacent, tcp="FAST", until=None, graph=True, 
                 tags=None):
        # Store graphing flag
        self._graph = graph
        # Data sets grouped by window title (data category)
        self._data = dict()
        # Create a network simulation with the given parameters
        n = network.Network(adjacent, tcp)
        # Run the simulation, and get the monitored values
        monitored = n.simulate(until)
        # For each monitored value
        for key, raw in monitored.items():
            # Extract the monitored title & identifying tags
            title, *tags = key.split(',')
            # If this is the first value of this category processed
            if title not in self._data.keys():
                # Create a new entry for its category
                self._data[title] = list()
            # Extract the time & value data from the raw data
            time, value = np.transpose(np.array(raw, dtype=float))
            # Scale time from nanoseconds to milliseconds
            time /= 1e6
            # Append the tagged data set to the data for this category
            self._data[title].append((tuple(tags), time, value))

    @property
    def titles(self):
        """Data categories.

        :return: list of data categories
        :rtype: [str]
        """
        return list(self._data.keys())

    def graph(self, title, datasets, legend, y_label, scale=1.0):
        """Graph a set of data sets.

        :param str title: graph title
        :param datasets: a list of tagged data sets given as (tags, x, y)
        :type datasets: [(tuple, ``numpy.ndarray``, ``numpy.ndarray``)]
        :param str legend: individual data set title for the plot legend
        :param str y_label: label for the plot\'s y axis
        :param float scale: y is multiplied by ``scale`` before graphing
        """
        # Get a new figure
        fig = plt.figure()
        # Set window title
        fig.suptitle(title)
        # Label the x axis 
        plt.xlabel("Time (ms)")
        # Label the y axis
        plt.ylabel(y_label)
        # For each tagged dataset
        for tags, x, y in datasets:
            y *= scale
            # Plot the data set, and make a legend entry for it
            plt.plot(x, y, label="{} {}".format(legend, ','.join(tags)))
        # Create the plot legend
        plt.legend()
        # If the graphing flag is set
        if self._graph:
            # Graph the data
            plt.show()
        else:
            # Save the plot
            plt.savefig(title + ".png")
        # Close all figures
        plt.close('all')

    def graph_all(self, tags=None):
        """Graph all data sets.

        If the ``tags`` parameter is given, graphical output for data
        sets whose title is a key in ``tags`` will be restricted to data
        sets with the specified tags.

        :param dict tags: dictionary mapping titles to data set tags
        :return: None
        """
        self.graph_titles(self._data.keys(), tags)

    def graph_titles(self, titles, tags=None):
        """Graph sets of data sets by category title.

        If the ``tags`` parameter is given, graphical output for data
        sets whose title is a key in ``tags`` will be restricted to data
        sets with the specified tags.

        :param list titles: titles of data categories to graph
        :param dict tags: dictionary mapping titles to data set tags
        :return: None
        """
        # For each title
        for title in titles:
            # Fetch the specified set of data sets
            datasets = self._data[title]
            # If this title exists in the tags dict
            if tags is not None and title in tags.keys():
                # Filter the data sets to those with the given tags
                datasets = [d for d in datasets if d[0] in tags[title]]
                tag_msg = " with tags {}".format(tags[title])
            else:
                tag_msg = str()
            # Check if there exist any data sets
            if not datasets:
                logger.warning(
                    "No data sets found for \"{}\"{}".format(title, tag_msg))
                # Do not graph null sets
                continue
            try:
                # Get the graphing parameters for this data category
                args = self.title_args[title]
            except KeyError:
                # If none were found, make empty legend & y axis labels
                args = [str(), str()]
            # Graph the data sets
            self.graph(title, datasets, *args)

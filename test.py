"""
.. module:: test
    :platform: Unix
    :synopsis: Testing utilites for the network simulator
"""

import enum
import heapq
import logging
import numpy as np
import simpy

from matplotlib import pyplot as plt

import gui
import process
import resources

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


class Graph:
    """Graphing class for generating simulation results.

    This class uses the :class:`Network` class to create and run a network
    simulation with the specified parameters, then groups the values
    monitored during that simulation by category (window title), assuming
    that all monitored identifiers have the form \"window title,tags\",
    where tags may itself be a comma-delimited string of integer values,
    such as an index, address, etc.  The instantiated object then provides
    various methods to graph/save the data sets using matplotlib.  The
    only obvious caveat when using the ``Graph`` class to run a simulation
    is that monitored data must be restricted to numerical values (anything
    that can be cast as a float) in order to actually produce graphical
    output.

    :param adjacent: an adjacency list of the format taken by :class:`Network`
    :type adjacent: ([((str, str), (int, int, int))], 
        [((str, str), ((int, int), (str, list)))]), or :class:`Case`
    :param str tcp: TCP specifier. Used iff adjacent is a :class:`Case`
    :param until: time or event to run the simulation until
    :type until: int or ``simpy.events.Event``
    """

    title_kwargs = {
        "Flow rate": {"legend": "flow", "y_label": "Mbps", "scale": 1000},
        "Round trip times": {"legend": "flow", "y_label": "ms", "scale": 1e-6},
        "Host rate": {"legend": "host", "y_label": "Mbps", "scale": 1000},
        "Link fill": {"legend": "link", "y_label": "packets"},
        "Dropped packets": {"legend": "link", "y_label": "packets"},
        "Link rate": {"legend": "link", "y_label": "Mbps", "scale": 1000},
        "Window size": {"legend": "flow", "y_label": "packets"},
        "Queuing delay": {"legend": "flow", "y_label": "ms", "scale": 1e-6}}
    """Maps graph title to :meth:`graph` keyword parameters."""

    def __init__(self, adjacent, tcp="FAST", until=None):
        # Data sets grouped by window title (data category)
        self._data = dict()
        # Create a network simulation with the given parameters
        n = Network(adjacent, tcp)
        # Run the simulation, and get the monitored values
        monitored = n.simulate(until)
        # For each monitored value
        for key, raw in monitored.items():
            # Extract the monitored title & identifying tags
            title, *tags = key.split(",")
            # Cast the tags as integers, and make them immutable
            tags = tuple(int(t) for t in tags)
            # If this is the first value of this category processed
            if title not in self._data.keys():
                # Create a new entry for its category
                self._data[title] = list()
            # Extract the time & value data from the raw data
            time, value = np.transpose(np.array(raw, dtype=float))
            # Scale time from nanoseconds to milliseconds
            time /= 1e6
            # Append the tagged data set to the data for this category
            self._data[title].append((tags, time, value))

    @property
    def titles(self):
        """Data categories.

        :return: list of data categories
        :rtype: [str]
        """
        return list(self._data.keys())

    def graph(self, title, datasets, legend=str(), y_label=str(), 
              scale=1.0, save=False):
        """Graph a set of data sets.

        :param str title: graph title
        :param datasets: a list of tagged data sets given as (tags, x, y)
        :type datasets: [(tuple, ``numpy.ndarray``, ``numpy.ndarray``)]
        :param str legend: legend label for each data set (e.g. \"flow\")
        :param str y_label: label for the plot\'s y axis
        :param float scale: y is multiplied by ``scale`` before graphing
        :param bool save: flag indicating whether to save or display figures
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
            y = y * scale
            # Plot the data set, and make a legend entry for it
            plt.plot(x, y, 
                label="{} {}".format(legend, ", ".join(str(t) for t in tags)))
        # Create the plot legend
        plt.legend()
        # If the graphing flag is set
        if not save:
            # Graph the data
            plt.show()
        else:
            # Save the plot
            plt.savefig(title + ".png")
        # Close all figures
        plt.close("all")

    def graph_all(self, tags=None, save=False):
        """Graph all data sets.

        If the ``tags`` parameter is given, graphical output for data
        sets whose title is a key in ``tags`` will be restricted to data
        sets with the specified tags.

        :param dict tags: dictionary mapping titles to data set tags
        :param bool save: flag indicating whether to save or display figures
        :return: None
        """
        self.graph_titles(self._data.keys(), tags, save)

    def graph_titles(self, titles, tags=None, save=False):
        """Graph sets of data sets by category title.

        If the ``tags`` parameter is given, graphical output for data
        sets whose title is a key in ``tags`` will be restricted to data
        sets with the specified tags.

        :param list titles: titles of data categories to graph
        :param dict tags: dictionary mapping titles to data set tags
        :param bool save: flag indicating whether to save or display figures
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
                args = self.title_kwargs[title]
            except KeyError:
                # If none were found, make empty legend & y axis labels
                args = dict()
            args["save"] = save
            # Graph the data sets
            self.graph(title, datasets, **args)


class Network:
    """This class encapsulates a network simulation.

    If ``adjacent`` is None, then a GUI for drawing a network appears.  The
    adjacency list should be formatted as [((src, dest), (capacity, buffer
    size, delay))], where src & dest are formatted as a string with a
    leading \"h\" or \"r\", specifying a host or router, followed by an
    integer id.  Flows should be given as [((src, dest), ((data, delay),
    (tcp, tcp_param)))], where src & dest are formatted as previously, but
    only host tags are allowed.  Link arguments are all in bits/bps, as
    is the data parameter for flows.  Delay is given in ns. The tcp 
    parameter is a string which determines which TCP algorithm to use;
    see :data:`process.Flow.allowed_tcp` for accepted TCP specifiers.
    The tcp_param parameter is a list containing the arguments for
    initializing the TCP object, and will change depending on the TCP
    algorithm specified.  See :mod:`process` for details of the currently
    implemented TCP algorithms.  Alternatively, adjacent may be a test
    case, as defined by the enumerated type :class:`Case`.  If so the
    ``tcp`` parameter may be a (valid) TCP specifier, used for all flows
    in the given test case.

    :param adjacent: adjacency lists of links & flows defining a network
    :type adjacent: ([((str, str), (int, int, int))], 
        [((str, str), ((int, int), (str, list)))]), or :class:`Case`
    :param str tcp: TCP specifier. Used iff adjacent is a :class:`Case`
    """

    def __init__(self, adjacent=None, tcp="FAST"):
        # Simulation environment
        self.env = resources.MonitoredEnvironment()
        # Table of hosts in this network
        self._hosts = dict()
        # Table of routers in this network
        self._routers = dict()
        # List of links in this network
        self._links = list()
        # List of flows in this network
        self._flows = list()

        # If no parameters were specified
        if adjacent is None:
            # Get edges & flows from GUI input
            adjacent = gui.draw()
        # Alternatively, if a test case was specified
        elif isinstance(adjacent, Case):
            # Build the corresponding adjacency list
            adjacent = TestCase.adjacent(adjacent, tcp)
        # Populate the network
        self._build_network(*adjacent)

    def _build_network(self, edges, flows):
        """Build a network of SimPy processes."""
        # Set of host addresses
        hosts = set()
        # Set of router addresses
        routers = set()

        # Parse the inputted addresses from the adjacency list
        for endpoints, _ in edges:
            # For each of the two endpoints
            for node in endpoints:
                # Add the tag's address to the appropriate set
                if node.startswith("h"):
                    hosts.add(int(node[1:]))
                else:
                    routers.add(int(node[1:]))

        # For each host address
        for addr in hosts:
            logger.debug("creating host {}".format(addr))
            # Make a new host with the given address
            self._hosts[addr] = process.Host(self.env, addr)
        # For each router address
        for addr in routers:
            logger.debug("creating router {}".format(addr))
            # Make a new router with the given address
            self._routers[addr] = process.Router(self.env, addr)

        # For each entry in the adjacency list
        for tags, parameters in edges:
            # Add link addr to parameters used to create a link
            parameters = list(parameters) + [len(self._links)]
            # Make a new link
            link = process.Link(self.env, *parameters)
            # Initialize a list of endpoints
            endpoints = list()
            # Logger parameters
            log_args = list()
            # Add an endpoint for each tag 
            for tag in tags:
                # Retrieve the address from this tag
                addr = int(tag[1:])
                # If it's a host tag
                if tag.startswith("h"):
                    # Append the host with the tagged address
                    endpoints.append(self._hosts[addr])
                    log_args.append("host")
                # Otherwise, it's a router
                else:
                    # Append the router with the tagged address
                    endpoints.append(self._routers[addr])
                    log_args.append("router")
                log_args.append(addr)

            logger.debug("connecting {} {} and {} {}".format(*log_args))
            # Connect the new link to its two endpoints
            link.connect(*endpoints)
            # Persist the new link
            self._links.append(link)

        # For each inputted flow
        for (src_tag, dest_tag), ((data, delay), (tcp, tcp_param)) in flows:
            # Get the source host
            src = self._hosts[int(src_tag[1])]
            # Get the destination address
            dest = int(dest_tag[1])
            logger.debug("creating flow between hosts {} and {}".format(
                src.addr, dest))
            # Create & persist the new flow
            self._flows.append(process.Flow(self.env, src, dest, data, delay,
                                            tcp, tcp_param))

    def simulate(self, until=None):
        """Run the initialized simulation.

        :param until: time or event to run the simulation until
        :type until: int or ``simpy.events.Event``
        :return: all monitored values
        :rtype: {str: [(int, object)]}
        """
        # Begin routing table initialization
        router_proc = \
            [self.env.process(r.begin()) for r in self._routers.values()]
        # Initialize packet generating processes for each flow
        flow_proc = [self.env.process(f.generate()) for f in self._flows]
        # If we didn't get a termination condition
        if until is None:
            # Terminate once all flows are finished transmitting data
            until = simpy.events.AllOf(self.env, 
                                       [f.finished for f in self._flows])
        # Run the simulation
        self.env.run(until=until)
        logger.info("all flows have terminated at time {}".format(
            self.env.now))
        # Retrieve monitored values
        values = self.env.monitored
        # Reset the environment
        self.env = resources.MonitoredEnvironment()
        # Return the monitored values
        return values


class TestCase:
    """Factory class for producing test case adjacency lists.

    See :meth:`adjacent` for details.
    """

    adjacencies = {
        Case.zero: ([(("h0", "h1"), (10000000, 512000, 10000000))], 
                    [(("h0", "h1"), (160000000, 1000000000))]),
        Case.one: ([(("h0", "r0"), (12500000, 512000, 10000000)),
                    (("r0", "r1"), (10000000, 512000, 10000000)),
                    (("r0", "r3"), (10000000, 512000, 10000000)),
                    (("r1", "r2"), (10000000, 512000, 10000000)),
                    (("r3", "r2"), (10000000, 512000, 10000000)),
                    (("r2", "h1"), (12500000, 512000, 10000000))],
                   [(("h0", "h1"), (160000000, 500000000))]),
        Case.two: ([(("r0", "r1"), (10000000, 1024000, 10000000)),
                    (("r1", "r2"), (10000000, 1024000, 10000000)),
                    (("r2", "r3"), (10000000, 1024000, 10000000)),
                    (("h0", "r0"), (12500000, 1024000, 10000000)),
                    (("h1", "r0"), (12500000, 1024000, 10000000)),
                    (("h2", "r2"), (12500000, 1024000, 10000000)),
                    (("h3", "r3"), (12500000, 1024000, 10000000)),
                    (("h4", "r1"), (12500000, 1024000, 10000000)),
                    (("h5", "r3"), (12500000, 1024000, 10000000))],
                   [(("h0", "h3"), (280000000, 500000000)),
                    (("h1", "h4"), (120000000, 10000000000)),
                    (("h2", "h5"), (240000000, 20000000000))])
    }
    """Edge & flow adjacency lists for each test case."""

    tcp_parameters = {
        Case.zero: {"FAST": [[1, 30000000, 45]], "Reno": [[1, 30000000]]}, 
        Case.one: {"FAST": [[1, 120000000, 20]], "Reno": [[1, 120000000]]},
        Case.two: {"FAST": [[1, 150000000, 6], 
                            [1, 90000000, 6],
                            [1, 90000000, 6]],
                   "Reno": [[1, 1500000000], 
                            [1, 90000000],
                            [1, 90000000]]}
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
            raise ValueError("invalid test case \"{}\"".format(case))
        if tcp not in process.Flow.allowed_tcp.keys():
            raise ValueError("invalid TCP specifier \"{}\"".format(tcp))

        # Initialize list of flows
        flows = list()
        for (tags, param), tcp_param in zip(cls.adjacencies[case][1], 
                                            cls.tcp_parameters[case][tcp]):
            # Add a flow with the proper TCP parameters
            flows.append((tags, (param, (tcp, tcp_param))))
        # Return adjacency lists of edges & initialized flows
        return cls.adjacencies[case][0], flows

"""
.. module:: network
    :platform: Unix
    :synopsis: This module implements a network simulator
"""

import logging
import simpy

import gui
import process
import test

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: calculate alpha for FAST TCP based on bottlenecks detected in the
#       input graph


class Network(object):
    """This class encapsulates a network simulation.

    If network is None, then a GUI for drawing a network appears.  The
    adjacency list should be formatted as [((src, dest), (capacity, buffer
    size, delay))], where src & dest are formatted as a string with a
    leading \"h\" or \"r\", specifying a host or router, followed by an
    integer id.  Flows should be given as [((src, dest), ((data, delay),
    (tcp, tcp_param)))], where src & dest are formatted as previously, but
    only host tags are allowed.  Link arguments are all in bits/bps, as
    is the data parameter for flows.  Delay is given in simulation time.
    The tcp parameter is a string which determines which TCP algorithm to
    use; accepted values are currently \"FAST\".  The tcp_param parameter
    is a list containing the arguments for initializing the TCP object,
    and will change depending on the TCP algorithm specified.  See
    :mod:`process` for details of the currently implemented TCP
    algorithms.

    :param adjacent: adjacency lists of links & flows defining a network
    :type adjacent`: ([((str, str), (int, int, int))], 
                      [((str, str), ((int, int), (str, list)))])
    """

    def __init__(self, adjacent=None):
        # Simulation environment
        self.env = test.MonitoredEnvironment()
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
                if node.startswith('h'):
                    hosts.add(int(node[1:]))
                else:
                    routers.add(int(node[1:]))

        # For each host address
        for addr in hosts:
            logger.info('creating host {}'.format(addr))
            # Make a new host with the given address
            self._hosts[addr] = process.Host(self.env, addr)
        # For each router address
        for addr in routers:
            logger.info('creating router {}'.format(addr))
            # Make a new router with the given address
            self._routers[addr] = process.Router(self.env, addr)

        # For each entry in the adjacency list
        for tags, parameters in edges:
            # Make a new link
            link = process.Link(self.env, *parameters)
            # Initialize a list of endpoints
            endpoints = list()

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

            logger.info('connecting {} {} and {} {}'.format(*log_args))
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
            logger.info('creating flow between hosts {} and {}'.format(
                src.addr, dest))
            # Create & persist the new flow
            self._flows.append(process.Flow(self.env, src, dest, data, delay,
                                            tcp, tcp_param))

    def simulate(self, until_=None):
        """Run the initialized simulation.

        :param int until_: time to run the simulation until
        :return: all monitored values
        :rtype: dict
        """
        # Initialize packet generating processes for each flow
        processes = [self.env.process(f.generate()) for f in self._flows]
        # Run the simulation
        self.env.run(until=until_)
        # Retrieve monitored values
        values = self.env.monitored
        # Reset the environment
        self.env = test.MonitoredEnvironment()

        return values

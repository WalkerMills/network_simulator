import logging
import simpy

import gui
import process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Network(object):

    def __init__(self, network=None):
        # Simulation environment
        self._env = simpy.Environment()
        # Table of hosts in this network
        self._hosts = dict()
        # Table of routers in this network
        self._routers = dict()
        # List of links in this network
        self._links = list()
        # List of flows in this network
        self._flows = list()

        # If no parameters were specified
        if network is None:
            # Get edges & flows from GUI input
            edges, flows = gui.draw()
        else:
            edges, flows = network
        # Populate the network
        self._build_network(edges, flows)

    def _build_network(self, edges, flows):
        # Set of host addresses
        hosts = set()
        # Set of router addresses
        routers = set()
        # Do some preprocessing on the given adjacency list & flows
        edges = [(l[0].split(','), l[1:]) for l in edges]
        flows = [(f[0].split(','), f[1:]) for f in flows]

        # Parse the inputted addresses from the adjacency list
        for endpoints, _ in edges:
            # For each of the two endpoints
            for node in endpoints:
                # Add the tag's address to the appropriate set
                if node.startswith('h'):
                    hosts.add(int(node[1]))
                else:
                    routers.add(int(node[1]))

        # For each host address
        for addr in hosts:
            logger.info('creating host {}'.format(addr))
            # Make a new host with the given address
            self._hosts[addr] = process.Host(self._env, addr)
        # For each router address
        for addr in routers:
            logger.info('creating router {}'.format(addr))
            # Make a new router with the given address
            self._routers[addr] = process.Router(self._env, addr)

        # For each entry in the adjacency list
        for tags, parameters in edges:
            # Make a new link
            link = process.Link(self._env, *parameters)
            # Initialize a list of endpoints
            endpoints = list()

            log_args = list()

            # Add an endpoint for each tag 
            for tag in tags:
                # Retrieve the address from this tag
                addr = int(tag[1])
                # If it's a host tag
                if tag.startswith("h"):
                    # Append the host with the tagged address
                    endpoints.append(self._hosts[addr])
                    log_args.append("host")
                # Otherwise, it's a router
                else:
                    # Append the router with the tagged address
                    endpoints.append(self._routers[int(tag[1])])
                    log_args.append("router")
                log_args.append(addr)

            logger.info('connecting {} {} and {} {}'.format(*log_args))
            # Connect the new link to its two endpoints
            link.connect(*endpoints)
            # Persist the new link
            self._links.append(link)

        # For each inputted flow
        for (src_tag, dest_tag), parameters in flows:
            # Get the source host
            src = self._hosts[int(src_tag[1])]
            # Get the destination address
            dest = int(dest_tag[1])
            logger.info('creating flow between hosts {} and {}'.format(
                src.addr, dest))
            # Create & persist the new flow
            self._flows.append(process.Flow(self._env, src, dest, *parameters))

    def simulate(self, until=None):
        # Initialize packet generating processes for each flow
        processes = [self._env.process(f.generate()) for f in self._flows]
        # Run the simulation
        self._env.run()

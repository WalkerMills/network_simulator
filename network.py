import simpy

import gui
import process


class Network(object):

    def __init__(self, network=None):
        if network is None:
            self._input = gui.draw()
        else:
            self._input = network

        # Simulation environment
        self.env = simpy.Environment()
        # List of hosts in this network
        self._hosts = list()
        # List of routers in this network
        self._routers = list()
        # List of links in this network
        self._links = list()
        # List of flows in this network
        self._flows = list()

    def build_network(self):
        # TODO: build a network from the input adjacency list & flows 
        hosts = set()
        routers = set()
        links, flows = self._input
        links = [(','.split(l[0]), l[1:]) for l in links]
        flows = [(','.split(f[0]), f[1:]) for f in flows]

        for endpoints, _ in links:
            for node in endpoints:
                if node.startswith('h'):
                    hosts.add(node)
                else:
                    routers.add(node)

        for host in sorted(hosts):
            addr = int(host[1])
            self._hosts.append(process.Host(self._env, addr))
        for router in sorted(routers):
            addr = int(router[1])
            self._routers.append(process.Router(self._env, addr))

    def simulate(self, until=None):
        # TODO: run the network simulation
        pass

"""
.. module:: test
    :platform: Unix
    :synopsis: Test cases (adjacency lists) for the network simulator.
"""

ZERO = ([(('h0', 'h1'), (8000000, 512000, 10))],
        [(('h0', 'h1'), (160000000, 32768, 100, 1000))])
"""Test case 0."""

ONE = ([(('h0', 'r0'), (100000000, 512000, 10)),
        (('r0', 'r1'), (80000000, 512000, 10)),
        (('r0', 'r3'), (80000000, 512000, 10)),
        (('r1', 'r2'), (80000000, 512000, 10)),
        (('r2', 'r3'), (80000000, 512000, 10)),
        (('h1', 'r2'), (100000000, 512000, 10))],
       [(('h0', 'h1'), (160000000, 32768, 100, 500))])
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
       [(('h0', 'h3'), (280000000, 32768, 200, 500)),
        (('h1', 'h4'), (120000000, 32768, 200, 10000)),
        (('h2', 'h5'), (240000000, 32768, 200, 20000))])
"""Test case 2."""

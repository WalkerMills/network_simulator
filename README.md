# CS 143 Network Simulator

Network simulator project for Caltech CS/EE 143.

## Dependencies

This project requires Python >= 3.3, SimPy >= 3.0, Sphinx >= 1.2, Tkinter,
NumPy, and Matplotlib. 

On Ubuntu:

```
# apt-get install python3 python3-simpy python3-sphinx python3-tk python3-numpy python3-matplotlib -y
```

However, by default Ubuntu uses Python 3.3, so if you follow the above
instructions, or your distro uses Python 3.3, you will also have to do:

```
# pip3 install enum34
```

in order to get the enum module, which was added in Python 3.4

On Arch Linux:

```
# pacman -S simpy python-sphinx python-pmw python-numpy python-matplotlib --noconfirm
```

## Documentation

This project is documented using Sphinx, a documentation generator.  To
build the documentation, simply run

```
$ make html
```

from the project root directory.  If you have a LaTeX compilation toolchain
installed, such as pdflatex, you may also run

```
$ make latexpdf
```

to build pdf documentation.  All documentation is output to the ./_build
directory.  The latest documentation (from the tip of develop) is also
available at

http://cs-143-network-simulator.readthedocs.org/en/latest/index.html

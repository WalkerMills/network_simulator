# CS 143 Network Simulator

Network simulator project for Caltech CS/EE 143.

## Dependencies

This project requires Python >= 3.2, SimPy >= 3.0, and Sphinx >= 1.2, and Tkinter

On Ubuntu:

$ sudo apt-get install python3 python3-simpy python3-sphinx python3-tk -y

On Arch Linux:

$ sudo pacman -S simpy python-sphinx python-pmw --noconfirm

## Documentation

This project is documented using Sphinx, a documentation generator.  To
build the documentation, simply run

$ make html

from the project root directory.  If you have a LaTeX compilation toolchain
installed, such as pdflatex, you may also run

$ make latexpdf

to build pdf documentation.  All documentation is output to the ./_build
directory.

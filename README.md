# CS 143 Network Simulator

Network simulator project for Caltech CS/EE 143.

## Dependencies

This project requires Python >= 3.3, SimPy >= 3.0, Tkinter, NumPy, and
Matplotlib. 

On Ubuntu < 14.04, you must first
```
# apt-get install python3 -y
```
Since versions prior to 14.04 do not come with Python 3 installed by default.
If your system uses Python 3.3 as its stable release of Python 3, you must also
```
# pip3 install enum34
```
in order to get the enum module, which was added in Python 3.4.  Finally, you
must install the Python module dependencies
```
# apt-get install python3-simpy python3-tk python3-numpy python3-matplotlib -y
```
On Arch Linux, Python 3.4 is the default Python version, so you must simply
```
# pacman -S simpy python-pmw python-numpy python-matplotlib --noconfirm
```

## Documentation

This project is documented using Sphinx, a documentation generator.  To
install Sphinx on Ubuntu
```
# apt-get install python3-sphinx -y
```
On Arch Linux
```
# pacman -S python-sphinx --noconfirm
```
To build the documentation, simply run
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

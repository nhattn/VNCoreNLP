# README

## [How do I Install GTK+ 3.0 on Ubuntu?](https://askubuntu.com/a/243636)

To use gtk2 or gtk3 apps you don't need to install anything. But, if you
want to develop (or even just compile) apps this is what you're looking for:

```
sudo apt-get install libgtk-3-dev
```

## [Installing Python and GTK on Windows](http://www.cs.dixie.edu/cs/1410/win_python_install.pdf)

If you have to limit yourself to Windows, here are some brief instructions
as to what you need to do to get it working.

Important note: this is already installed and working on the lab machines.
If you wish to install it on your personal machine, you are responsible
for getting it to work. The instructor and labbies are not responsible
for installing it for you, and failing to get it working is not an acceptable
excuse for late or missing work. If you want to use your own machine, the
responsibility is yours and yours alone to make sure that you have a
working environment.

- Install Python 2.7.2 or greater for Windows from the Python website. The [link](https://python.org) is copied here for your
convenience:
> \- For 32-bit and 64-bit Windows: [Python 2.7.2 Windows Installer](https://www.python.org/ftp/python/2.7.2/python-2.7.2.msi)
Note that this will install the 32-bit version of Python, even for 64-bit Windows. This is the correct thing
to do. The 64-bit installer causes problems with some of the libraries that we will need.
The installer is a standard MSI file. Double click it after downloading and the installer will start.
- Install the PyGTK all-in-one installer from the [PyGTK website](http://pygtk.org/downloads.html), which includes GTK+, PyGTK, PyCairo, and
PyGObject: [PyGTK all-in-one installer](http://ftp.gnome.org/pub/GNOME/binaries/win32/pygtk/2.24/pygtk-all-in-one-2.24.1.win32-py2.7.msi)
The installer is a standard MSI file. Double click it after downloading and the installer will start.

To check if everything is working, open an interactive python shell and type this in:

```python
import gtk
window = gtk.Window()
window.set_title("PyGTK Test Window")
window.connect("destroy", gtk.main_quit)
window.show_all()
gtk.main()
```

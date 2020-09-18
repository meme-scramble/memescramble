Welcome to MemeScramble

###################################################################################

Coded using python 3.8.3 in the anaconda3 distribution 
The following packages were used:
numpy 1.18.2
matplotlib 3.2.2
PIL 6.2.2
scipy 1.5.0
tkinter 8.6
copy
random
cx_Freeze 6.2 (for compiling only)

###################################################################################

This program edits images. The following image file types are supported:
.bmp
.dds
.exif
.gif
.jpg, .jpeg
.png
.tif, .tiff

###################################################################################

To run the program double-click the executable file memescramble.exe (we reccomend making a shortcut to this file).

WARNING: The program must be exited using the "Exit" button rather than the "X" in the upper-right corer. If the "X" is used,
the window will close, but the program will still be running in the background.

Misson:
This program is intended to help meme artists disguise their memes so they can sneak past machine learning algorithms used to 
identify memes.  There are 9 different types of transformations that can be applied to an image. The intent of these transformations
is to make the meme identifiable to a human being, while fooling a computer algorithm. It is still undetermined how effective each of
these transformations will be, as we do not have access to the algorithms used to identify memes.

Main menu buttons:
Browse files - use this to load an image from anywhere on your pc
Start Over - If transformations have been applied, this button resets the preview to the unaltered image, and also clears the history of
	undo and redo operations.
Undo - undo a transformation
Redo - redo a transformation after it has been undone
Save Result As - Save the resulting image anywhere on your pc
Reset Settings to Defaults - Reset all sliders controlling the transformations to the default settings they start with upon program startup
Apply Random Transformation - Applies a random number of transformations with random intensities.
Exit - Exit the program. Note, the program must be exited with this button for the program to stop completely. If the "X" in the upper-right
	corner of the window is used instead, the window will close, but the program will keep running in the background.

Source code:
In the source_code directory, memescramble.py is the python file containing all the source code for the program. This can be executed 
in a python interpreter to run the program the same way the program would run from the executable.  The script setup.py is used to compile
the program. The program is compiled with the following command:

python setup.py build_exe --excludes=matplotlib.tests,numpy.random._examples

After compiling, the user must manually create an empty folder called temp_images inside the resulting build directory.
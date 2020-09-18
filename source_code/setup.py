"""
Compiled with cx_Freeze v6.2
compile with the command:
   python setup.py build_exe --excludes=matplotlib.tests,numpy.random._examples
   
After compiling:
    -make empty folder in exe directory called temp_images
    -create folder in exe directory called source_code and copy memescramble.py and setup.py into it.
"""

import cx_Freeze
import sys
import matplotlib

base = None

if sys.platform == 'win32':
    base = "Win32GUI"

executables = [cx_Freeze.Executable("memescramble.py", base=base, icon="memescramble.ico")]

cx_Freeze.setup(
    name = "meme_scrambler",
    options = {"build_exe": {"packages":["tkinter","matplotlib","scipy"],"include_files":["memescramble.ico"]}},
    version = "0.1",
    description = "",
    executables = executables
    )
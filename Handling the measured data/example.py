PATH_LIB = '/home/florian/code/libs/'

import os, sys
sys.path.append(os.path.dirname(PATH_LIB))
import filehandling as fh
import labview_import

dfiles = fh.get_files(filetype=labview_import.Labview_file_eh, directory='data/')
dfiles.sort(key=lambda dfile: dfile.source)
dfilesd = {dfile.basename : dfile for dfile in dfiles}
print(dfilesd.keys())

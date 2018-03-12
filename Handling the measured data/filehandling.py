#!/usr/bin/env python
'''
This library handles all file operations.
'''
import os


class Datafile(object):
    '''
    Class for data files.
    '''
    def __init__(self, source):
        self.source = source
        self.path, self.basename, self.extension = extract_name(source)
        self.desc = 'Generic file.'
        if not os.path.isfile(source):
            raise IOError("Input file does not exist or is not a valid file.")

    def get_info(self):
        '''
        Returns a list with some information about the Datafile
        '''
        return (self.source, self.path, self.basename, self.extension,
                self.desc)

    def set_desc(self, desc):
        '''
        Set a human readable description of the file
        '''
        self.desc = desc


def extract_name(source):
    '''
    Extracts the path, the filename and the extension of a source string and
    returns a list of these three.
    '''
    return (os.path.dirname(source),
            os.path.split(os.path.splitext(source)[0])[1],
            os.path.splitext(source))


def get_files(filetype=Datafile, directory=os.curdir,
              file_extension='.txt'):
    '''
    Function to return a list of all the files in the folder given by directory
    with the proper extension given by file_extension.
    '''
    files = []
    #print file_extension, 'files found in the directory', directory +':'
    for filename in os.listdir(directory):
        if (os.path.isfile(os.path.join(directory, filename)) and
            os.path.splitext(filename)[1] == file_extension):
            files.append(filetype(os.path.join(directory, filename)))
    return files

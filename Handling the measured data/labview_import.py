import filehandling as fh
import re
import numpy as np
import time
from scipy.interpolate import griddata
import matplotlib.pyplot as pl


class Labview_file_eh(fh.Datafile):
    def __init__(self, source):
        super(Labview_file_eh, self).__init__(source)
        self.traces = []
        self._read()

    def _read(self):
        '''Read in the data source and store them into self.traces.'''
        with open(self.source) as file_in:
            file_content = file_in.read()

        datasets = re.split('Measurement', file_content)
        del file_content
        datasets.pop(0)
        if datasets[-1][-2:] != '\r\n':
            datasets[-1] = datasets[-1] + '\r\n'  #  Make the last datasets equal to the others; necessary not to fail the reading
        for dataset in datasets:
            self.traces.append(Trace(dataset))
        if len(self.traces) > 1:
            self.map = Map(self.traces)


class Trace(object):
    def __init__(self, dataset):
        self._read(dataset)

    def _read(self, dataset):
        #Split content of a dataset in paragraphs for further processing
        groups = re.split("\r\n\r\n", dataset)
        header = groups[0:-1]
        data = groups[-1]

        #Split the data section into a header and a footer
        datahead, data = re.split("\r\n", data, maxsplit=1)
        datahead = re.split("\t", datahead)
        data = re.split("\s*", data)
        data.pop(-1)
        try:
            data = np.array([float(i) for i in data]).reshape(-1, len(datahead))
        except ValueError:  # Try to corrigate invalid value
            data_list = []
            for i in data:
                try:
                    entry = float(i)
                except ValueError:
                    pattern = re.compile('(?P<number>\d*)')
                    match = pattern.search(i)
                    try:
                        entry = float(match.group('number'))
                    except ValueError:  # If everything fails, set it to 0
                        entry = 0
                data_list.append(entry)
            data = np.array(data_list).reshape(-1, len(datahead))

        #Extract the channels' names and units from the datahead
        #By convention: Take last occurence of (string) and assume string = unit
        channels = []
        for channel in datahead:
            splits = channel.split(' ')
            splits.reverse()
            for i in range(len(splits)):
                if splits[i][0] == '(' and splits[i][-1] == ')':
                    unit = splits.pop(i)[1:-1]
                    break
                else:
                    unit = ''
            splits.reverse()
            channel = ' '.join(splits)
            channels.append((channel, unit))

        #Determine the type of each block in the header section
        #Date
        pattern = re.compile('(?P<date>\d{2}.\d{2}.\d{4}) at (?P<time>\d{2}:\d{2})')
        match = pattern.search(dataset)
        date = match.group('date') + ' ' + match.group('time')
        date = time.strptime(date, '%d.%m.%Y %H:%M')

        #Stepping Channels
        pattern = re.compile('Stepping Channel:')
        steps = []
        for block in header:
            if pattern.match(block):
                m = re.search('Stepping Channel: (?P<key>.*?)\r', block)
                steps.append(m.group('key'))

        #Parameters
        parameters = []

        #Voltage Controls
        pattern = re.compile('Voltage Controls:')
        for block in header:
            if pattern.match(block):
                block = re.split('\r\n', block)
                block.pop(0)
                for line in block:
                    m = re.search('(?P<key>.*?): (?P<value>.*?) (?P<unit>.*?) (?P<rest>.*)',
                              line)
                    m2 = re.search('Freq: (?P<value>.*?) (?P<unit>.*)', m.group('rest'))
                    parameters.append((m.group('key'), float(m.group('value')), m.group('unit')))
                    if m2:
                        parameters.append((m.group('key'), float(m2.group('value')), m2.group('unit')))

        #Magnetic Field
        pattern = re.compile('Magnetic Field:')
        for block in header:
            if pattern.match(block):
                m = re.search('Magnetic Field: (?P<value>.*?) (?P<unit>.*?)\r', block)
                if 'Magnetic Field' not in [channel[0] for channel in channels]:
                    parameters.append(('Magnetic Field', float(m.group('value')), m.group('unit')))

        #Add all the gathered information into object variables
        self.channels = channels
        self.date = date
        steps_values = []
        for parameter in parameters:
            if parameter[0] in steps:
                steps_values.append(parameter)
        self.steps = steps_values
        self.parameters = parameters
        self.data = data


class Map(object):
    '''This object builds a 2D Map from a list of traces.
    self.date           :       Tuple of min and max time
    self.data           :       XYZ data'''
    def __init__(self, traces, xchannel=0, ychannel=2, zchannel=1):
        data = np.array([])
        date = []
        #Check if data is ordered from small to high parameters,
        #otherwise make it happen
        #
        #However, so far no arbitrarily sorted data possible
        if traces[0].parameters[ychannel][1] > traces[1].parameters[ychannel][1]:
            traces.reverse()
        for trace in traces:
            #Date extraction
            date.append(time.mktime(trace.date))
            #Data extraction
            x = trace.data[:, xchannel]
            z = trace.data[:, zchannel]
            y_value = trace.parameters[ychannel][1]
            y = np.ones(len(x)) * y_value
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            z = z.reshape(-1, 1)
            xyz = np.append(x, y, axis=1)
            xyz = np.append(xyz, z, axis=1)
            data = np.append(data, xyz)
        self.date = (time.localtime(min(date)), time.localtime(max(date)))
        data = data.reshape(-1, 3)
        self.data = data
        x_uvals = np.unique(data[:, 0])
        y_uvals = np.unique(data[:, 1])
        if len(x_uvals) * len(y_uvals) == len(data):
            self.shape = (len(y_uvals), len(x_uvals))
        else:
            self.shape = (0, 0)
        self.channels = [traces[0].channels[xchannel], (traces[0].parameters[ychannel][0], traces[0].parameters[ychannel][2]), traces[0].channels[zchannel]]

    def interpolate(self, shape, method='nearest'):
        '''Build a rectified data array by interpolating missing values.'''
        x = self.data[:, 0]
        y = self.data[:, 1]
        z = self.data[:, 2]
        grid = shape
        grid_x, grid_y = (
        np.mgrid[x.min():x.max():grid[0] * 1j, y.min():y.max():grid[1] * 1j])

        #Interpolate Z-data
        method = 'nearest'
        Z = griddata((x, y), z, (grid_x, grid_y), method=method)
        self.griddata = (grid_x, grid_y, Z)

    def rectify(self, shape=None):
        '''Build a rectified masked data array.'''
        x = self.data[:, 0]
        y = self.data[:, 1]
        z = self.data[:, 2]
        xi = np.unique(x)
        yi = np.unique(y)
        if not shape:
            shape = (len(yi), len(xi))
        X, Y = np.meshgrid(xi, yi)
        xmax = xi.max()
        xmin = xi.min()
        ymax = yi.max()
        ymin = yi.min()
        step_x = (xmax - xmin) / (len(xi) - 1)
        step_y = (ymax - ymin) / (len(yi) - 1)
        coordinates = [(x[i], y[i], z[i]) for i in range(len(x))]
        #False is unmasked, True is masked
        mask = np.ones(shape, dtype=np.bool)
        z_rect = np.zeros(shape)
        for xii, yii, zii in coordinates:
            pos_x = int(round((xii - xmin) / step_x))
            pos_y = int(round((yii - ymin) / step_y))
            mask[pos_y, pos_x] = False
            z_rect[pos_y, pos_x] = zii
        z_masked = np.ma.array(z_rect, mask=mask)
        self.rectdata = (X, Y, z_masked)

    def rectify_triangle(self, shape=None):
        '''Build a rectified masked data array from an inequally x-stepped array.
        y-step is supposed to be equally stepped.'''
        x = self.data[:, 0]
        y = self.data[:, 1]
        z = self.data[:, 2]
        # Assume y is right and determine the length of xi only by accounting the occurence y.max depending on which is larger
        xi = np.linspace(x.min(), x.max(), len(x[y==y.max()]))
        yi = np.unique(y)
        if not shape:
            shape = (len(yi), len(xi))
        X, Y = np.meshgrid(xi, yi)

        z_rect = np.zeros(shape)
        lens = np.array([len(x[y==yii]) for yii in yi])
        for n in xrange(len(lens)):
            position = -lens[-n - 1:].sum()
            position_end = position + lens[-n - 1] - 1
            value = x[position]
            idx = (np.abs(xi-value)).argmin()
            if n == 0:
                z_rect[n, idx - lens[-n-1] + 1:idx + 1] = z[position:]
            else:
                z_rect[n, idx - lens[-n-1] + 1:idx + 1] = z[position:position_end + 1]
        z_rect = np.fliplr(z_rect)
        mask = (z_rect == 0)
        z_masked = np.ma.array(z_rect, mask=mask)
        self.rectdata = (X, Y, z_masked)

    def plot(self, data=None, shape=None, clim=None, diff=None):
        if not data:
            data = self.data
        if not shape:
            shape = self.shape
        x = data[:, 0].reshape(shape)
        y = data[:, 1].reshape(shape)
        z = data[:, 2].reshape(shape)
        y = np.flipud(y)
        z = np.flipud(z)
        if diff is not None:
            z = np.diff(z, axis=diff)
        fig = pl.figure()
        ax = pl.subplot(111)
        img = pl.imshow(z, aspect='auto', extent=[x[0,0], x[0, -1], y[-1, 0], y[0, 0]], cmap=pl.get_cmap('Blues_r'), interpolation='none')
        ax.set_xlabel(self.channels[0][0] + ' in ' + self.channels[0][1])
        ax.set_ylabel(self.channels[1][0] + ' in ' + self.channels[1][1])
        c = pl.colorbar()
        c.set_label(self.channels[2][0] + ' in ' + self.channels[2][1])
        if clim:
            pl.clim(clim)
        pl.minorticks_on()
        return fig, ax, img, c


if __name__ == "__main__":
    dfile = Labview_file_eh("data/current/July 25 -- Delta vs Power fine_1.txt")
    print dfile.traces[0].data

# -*- coding: utf-8 -*-

__all__ = ['Experiment']

import xml.etree.ElementTree as ET
from pathlib import Path
import joblib
import pandas as pd
import numpy as np


class Experiment(object):
    """[summary]

    Parameters
    ----------
    object : [type]
        [description]
    """
    def __init__(self, path,
                 scan_init_file='InitValues.csv',
                 scan_setup_file='ScanLists.csv',
                 scan_value_file='ScanValues.csv',
                 exp_setting_file='SettingsV3.xml'):
        """[summary]

        Parameters
        ----------
        path : str
            path to the experiment data file
        scan_init_file : str, optional
            file name of scan initial settings, by default 'InitValues.csv'
        scan_setup_file : str, optional
            file name of scan setup, by default 'ScanLists.csv'
        scan_value_file : str, optional
            file name of scan values, by default 'ScanValues.csv'
        exp_setting_file : str, optional
            file name of experiment settings, by default 'SettingsV3.xml'
        """
        self.__path = Path(path)
        if not self.__path.joinpath('assets').exists():
            self.__path.joinpath('assets').mkdir(parents=True, exist_ok=False)
        self._setting_assembly(scan_init_file, scan_setup_file,
                               scan_value_file, exp_setting_file)

        self.__r = self._get_data()

    @property
    def path(self):
        return self.__path

    @property
    def scan_init(self):
        return self.__scan_init

    @property
    def scan_setup(self):
        return self.__scan_setup

    @property
    def scan_value(self):
        return self.__scan_value

    @property
    def scan_size(self):
        return self.__scan_size

    @property
    def number_of_readout(self):
        return self.__number_of_readout

    @property
    def average_axis(self):
        return self.__average_axis

    @property
    def data(self):
        return self.__r

    def average(self, average_axis=None):

        if average_axis is None:
            average_axis = self.__average_axis

        averaged_data = np.mean(self.data, axis=average_axis)

        return averaged_data

    def mean_mag(self, decibel=True):
        if decibel:
            mag = 20*np.log10(np.linalg.norm(self.average(), axis=1))
        else:
            mag = np.linalg.norm(self.average(), axis=1)

        return mag

    def mean_phase(self, deg=False):
        complex_data = [[None] for _ in range(self.__number_of_readout)]
        for res in range(self.__number_of_readout):
            complex_data[res] = self.average()[res][0] +1j*self.average()[res][1]

        if deg:
            phz = np.angle(complex_data, deg=True)
        else:
            phz = np.angle(complex_data)

        return phz

    def _setting_assembly(self, scan_init_file, scan_setup_file,
                          scan_value_file, exp_setting_file):
        if self.__path.joinpath('assets/settings.joblib').exists():
            with open(self.__path.joinpath('assets/settings.joblib'), 'rb') as handle:
                setting_ensemble = joblib.load(handle)

            self.__scan_init = setting_ensemble['scan_init']
            self.__scan_setup = setting_ensemble['scan_setup']
            self.__scan_value = setting_ensemble['scan_value']
            self.__scan_size = setting_ensemble['scan_size']
            self.__number_of_readout = setting_ensemble['number_of_readout']
            self.__average_axis = setting_ensemble['average_axis']

        else:
            self.__scan_init = pd.read_csv(
                self.__path.joinpath(scan_init_file),
                index_col=0, header=None,
                float_precision='round_trip')
            self.__scan_init.index.name = None
            self.__scan_init.columns = self.__scan_init.columns - 1
            # -1 is to fix name mismatch

            self.__scan_setup = pd.read_csv(
                self.__path.joinpath(scan_setup_file),
                index_col=None, header=None,
                float_precision='round_trip')
            self.__scan_setup.index = ['Enabled', 'Target', 'Parameter', 'Scan #', 'Object #',
                                       'Start', 'Stop', '# of Step']
            self.__disabled_object = np.nonzero(self.__scan_setup.loc[
                'Enabled', :].str.contains('FALSE').to_numpy())[0]
            self.__scan_setup = self.__scan_setup.drop(self.__disabled_object, axis=1)
            self.__scan_setup.columns = range(len(self.__scan_setup.columns))

            self.__scan_size = self.__scan_setup.iloc[[-1]].to_numpy().astype(int)[0]

            self.__average_axis = tuple(np.nonzero(self.__scan_setup.loc[
                'Target', :].str.contains('Repeat').to_numpy())[0]+2)

            self.__scan_value_raw = pd.read_csv(
                self.__path.joinpath(scan_value_file),
                index_col=None, header=None, skiprows=self.__disabled_object,
                float_precision='round_trip').to_numpy()
            
            tree = ET.parse(self.__path.joinpath(exp_setting_file))
            root = tree.getroot()
            for array in root.iter('{http://www.ni.com/LVData}Array'):
                if array.find('{http://www.ni.com/LVData}Name').text == 'HeteroFreq':
                    self.__number_of_readout = int(
                        array.findall('{http://www.ni.com/LVData}Dimsize')[1].text)
            for ew_element in root.iter('{http://www.ni.com/LVData}EW'):
                if ew_element.find('{http://www.ni.com/LVData}Val').text == '5':
                    for u32 in root.iter('{http://www.ni.com/LVData}U32'):
                        if u32.find('{http://www.ni.com/LVData}Name').text == '#rec':
                            self.__scan_size[0] = int(u32.find('{http://www.ni.com/LVData}Val').text)
                            self.__scan_value_raw[0, :self.__scan_size[0]] = np.arange(self.__scan_size[0])

            self.__scan_value = [np.zeros(dim) for dim in self.__scan_size]
            for dim in range(len(self.__scan_size)):
                target = self.__scan_setup.loc[['Target', 'Parameter'], dim].str.cat(sep=' ')
                if target.lower() in self.__scan_init.index.str.lower():
                    init_val_row = self.__scan_init.index.str.lower().get_loc(target.lower())
                    init_val_col = int(self.__scan_setup.loc['Object #', dim])
                    init_val = self.__scan_init.iloc[init_val_row, init_val_col]
                else:
                    init_val = 0
                self.__scan_value[dim] = self.__scan_value_raw[dim, :self.__scan_size[dim]] + init_val
                self.__scan_value[dim].round(decimals=6)

            setting_ensemble = dict(scan_init=self.__scan_init,
                                    scan_setup=self.__scan_setup,
                                    scan_value=self.__scan_value,
                                    scan_size=self.__scan_size,
                                    number_of_readout=self.__number_of_readout,
                                    average_axis=self.__average_axis)

            with open(self.__path.joinpath('assets/settings.joblib'), 'wb') as handle:
                joblib.dump(setting_ensemble, handle)

    def _get_data(self):
        if self.__path.joinpath('assets/data.joblib').exists():
            self.__r = joblib.load(self.__path.joinpath('assets/data.joblib'))

        else:
            self.__r = [[np.zeros(self.__scan_size) for _ in range(2)] for _ in range(self.__number_of_readout)]
            # size of data array would be (# of readout)*(quadrature count)*(scan array)
            if len(self.__scan_size) < 3:
                try:
                    pd.read_csv(self.__path.joinpath(f'R{self.__number_of_readout-1}I.csv'))
                except FileNotFoundError:
                    raise FileNotFoundError('Experiment not completed.')

                for res in range(self.__number_of_readout):
                    self.__r[res][0] = pd.read_csv(
                        self.__path.joinpath(f'R{res}I.csv'),
                        index_col=None, float_precision='round_trip',
                        header=None).T.to_numpy()
                    self.__r[res][1] = pd.read_csv(
                        self.__path.joinpath(f'R{res}Q.csv'),
                        index_col=None, float_precision='round_trip',
                        header=None).T.to_numpy()
            elif len(self.__scan_size) == 3:
                try:
                    pd.read_csv(self.__path.joinpath(f'R{self.__number_of_readout-1}I_{self.__scan_size[-1]-1}.csv'))
                except FileNotFoundError:
                    raise FileNotFoundError(f'Experiment not completed. {self.__scan_size[-1]} experiments expected.')

                for res in range(self.__number_of_readout):
                    for scan in range(self.__scan_size[-1]):
                        self.__r[res][0][:, :, scan] = pd.read_csv(
                            self.__path.joinpath(f'R{res}I_{scan}.csv'),
                            index_col=None, float_precision='round_trip',
                            header=None).T.to_numpy()

                        self.__r[res][1][:, :, scan] = pd.read_csv(
                            self.__path.joinpath(f'R{res}Q_{scan}.csv'),
                            index_col=None, float_precision='round_trip',
                            header=None).T.to_numpy()

            joblib.dump(self.__r, self.__path.joinpath('assets/data.joblib'))

        return self.__r

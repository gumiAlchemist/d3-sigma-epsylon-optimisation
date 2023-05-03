import pandas as pd
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openmm import app
from dftd3.interface import RationalDampingParam, DispersionModel
import shutil
import scipy
import openmm
from onlyEnergy import *
from Edissotiation_for_all import *
elements = ['F', 'C', 'H', 'Cl', 'N', 'O', 'S', 'Br', 'I']
nums = [2.9, 0.06, 3.55, 0.07, 2.46, 0.03, 3.4, 0.3, 3.25, 0.17, 3.0, 0.17, 3.6, 0.355, 3.47, 0.47, 3.55, 0.58, 1.0, 0.9171, 0.3385, 2.883]
ns6, ns8, na1, na2 = nums[-4], nums[-3], nums[-2], nums[-1]
sigma = {el: nums[2 * i]/10 for i, el in enumerate(elements)}
epsilon = {el: nums[2 * i+1]*4.184 for i, el in enumerate(elements)}
numatoms = {'F':9, 'C':6, 'H':1, 'Cl':17, 'N':7, 'O':8, 'S':16, 'Br':35, 'I':53}

excel_data = pd.read_excel('/mnt/c/ПРоект/table.xlsx')
data = pd.DataFrame(excel_data, columns=['name', 'mol1', 'mol2', 'E dissotiation'])
t=list(data['name'])

def calc_Error(file):
    Error=0
    E_calc = Edissotiation(file, sigma, epsilon, ns6, ns8, na1, na2, forcefield)
    line = data[data['name'] == file[:-4]]
    Etable=float(line['E dissotiation'].loc[line.index[0]])
    Error += abs(E_calc - Etable)
    return Error

for fil in t:
    file=fil+'.pdb'
    my_file = open("/mnt/c/ПРоект/вывод/Error.txt", "a")
    print(file[:-4], str(calc_Error(file)), file=my_file)
    my_file.close()
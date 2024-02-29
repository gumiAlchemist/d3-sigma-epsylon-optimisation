import numpy as np
import pandas as pd
from datetime import date
import scipy
from openmm.app import ForceField, PDBReporter, Element
from openmm import LangevinMiddleIntegrator, CustomNonbondedForce
from openmm.app import PDBFile, ForceField, Modeller, Simulation, Topology
from openmm.unit import kilocalorie_per_mole, picosecond, picoseconds, kelvin, nanometer
from calc_energies_ds_end_another_form import OPLS_LJ, to_openmm, change_sigma_n_epsilon
from FilesProcessing import parse_pdb, Molecule
from typing import Dict, List, Tuple
from dftd3.interface import RationalDampingParam, DispersionModel, Structure
import os
import sys
from datetime import datetime
sys.path.insert(0, '/mnt/c/ПРоект/git_xml/OGEN-master/scripts')

elements=['AR']
sigma={'AR': 0.0}
epsilon={'AR':0.0}

NM_TO_BOHR=0.052917721090380

nums=[0.5302*100, 1273*4.184/10]
sigma['AR'], epsilon['AR']=nums
#print(sigma, epsilon)

pdbs=[os.path.join('/mnt/c/ПРоект/NCfailed/optimcheck/pdb2th/', f) for f in os.listdir('/mnt/c/ПРоект/NCfailed/optimcheck/pdb2th/')]
xmls=['/mnt/c/ПРоект/NCfailed/optimcheck/xml2thnum/ArAr.xml']
change_sigma_n_epsilon('/mnt/c/ПРоект/NCfailed/optimcheck/xml2thwrd/', '/mnt/c/ПРоект/NCfailed/optimcheck/xml2thnum/', sigma, epsilon, elements)
xml = ForceField(*xmls)
s6, s8, a1, a2=[1.0, 0.4145, 1.2177, 4.8593] #bj pbe0
#s6, s8, a1, a2=[1.0, 0.9171, 0.3385, 2.883]

excel_data = pd.read_excel('/mnt/c/ПРоект/NCfailed/optimcheck/ArArEdiss.xlsx')
data = pd.DataFrame(excel_data, columns=['R, A', 'E, kcal'])


def calc_Eopmm(sigma, epsilon, s6, s8, a1, a2, numrs, topology, positions, ff, name):
    #print(topology, positions)
    model = Modeller(topology, positions)
    model.addExtraParticles(ff)
    sys = ff.createSystem(model.topology)
    sys = OPLS_LJ(sys, sigma, epsilon)
    integrator = LangevinMiddleIntegrator(
        300 * kelvin,
        1 / picosecond,
        0.004 * picoseconds
    )
    sim = Simulation(model.topology, sys, integrator)
    sim.context.setPositions(model.positions)
    E = sim.context.getState(getEnergy=True)
    OpenmmEnergy =E.getPotentialEnergy() / kilocalorie_per_mole
    posit=np.array(positions)/NM_TO_BOHR
    numbers=np.array(numrs, dtype=int)
    mod = DispersionModel (numbers, posit)
    res = mod.get_dispersion(RationalDampingParam(s6=s6, s8=s8, a1=a1, a2=a2), grad=False)
    EpyD3=(res.get("energy"))*((4.359744722207185e-18)*6.02e23*(1e-3)/4.198) # ккал/моль  #энергия хартри(википедия)
    
    return OpenmmEnergy+EpyD3

def calc_diss(sigma, epsilon, pdbs, xml):
    #energies={}
    
    Eerror=[]
    for pdb in pdbs:
        #print(pdb)
        name, _ = os.path.splitext(os.path.basename(pdb))
        
        mol = PDBFile(pdb)
        _, components, _ = parse_pdb(pdb)
        topologies = []
        positions = []
        numrs=[]
        for component in components:
            t, p, num = to_openmm(component)
            topologies.append(t)
            positions.append(p)
            numrs.append(num)
        dim_posit=mol.getPositions()/nanometer
        dim_numrs=[a.element.atomic_number for a in mol.topology.atoms()]
        e = calc_Eopmm(sigma, epsilon, s6, s8, a1, a2, dim_numrs, mol.topology, dim_posit,
                                         xml, name)
        #energies[name] = e
        #print('dim1111', e)
        diss_energy = e
        for i, (t, p, num) in enumerate(zip(topologies, positions, numrs)):
            component_name = '%s_mol%02d' % (name, i + 1)
            e = calc_Eopmm(sigma, epsilon, s6, s8, a1, a2, num, t, p, xml,
                                component_name)
            #print('comp2221111', e)
            #energies[component_name] = e
            diss_energy -= e
        #energies['%s_diss' % name] = diss_energy
        

        rfile=pdb.split('/')[-1][4:-4]
        line = data[round(data['R, A'], 3) == int(rfile)/10]
        Etable=float(line['E, kcal'].loc[line.index[0]])
        #print(abs(Etable-(diss_energy/(1.380649*6.02214076*4.184*(10**-3)))))
        Eerror.append(abs(Etable-diss_energy))
    return np.array(Eerror)





def calc_error(nums):
    sigma['AR'], epsilon['AR']=nums
    change_sigma_n_epsilon('/mnt/c/ПРоект/NCfailed/optimcheck/xml2thwrd/', '/mnt/c/ПРоект/NCfailed/optimcheck/xml2thnum/', sigma, epsilon, elements)
    xml = ForceField(*xmls)
    Error=calc_diss(sigma, epsilon, pdbs, xml)
    Errors=sum(Error)
    current_datetime = datetime.now().strftime("%Y-%m-%d")
    str_current_datetime = str(current_datetime)
    file_name = "/mnt/c/ПРоект/NCfailed/optimcheck/thirdARARsteptktj"+str_current_datetime+".txt"
    file = open(file_name, 'a')
    print(Errors, sigma, epsilon, s6, s8, a1, a2, file=file)
    file.close()
    return Errors

#result = scipy.optimize.dual_annealing(calc_error, bounds=[(0, 10000), (0, 50000)])
result = scipy.optimize.minimize(calc_error, np.array(nums), method = 'Nelder-Mead', options={'disp': True})
print(result.x)


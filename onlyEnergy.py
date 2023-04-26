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
elements = ['F', 'C', 'H', 'Cl', 'N', 'O', 'S', 'Br', 'I']
nums = [2.9, 0.06, 3.55, 0.07, 2.46, 0.03, 3.4, 0.3, 3.25, 0.17, 3.0, 0.17, 3.6, 0.355, 3.47, 0.47, 3.55, 0.58, 1.0, 0.9171, 0.3385, 2.883]
ns6, ns8, na1, na2 = nums[-4], nums[-3], nums[-2], nums[-1]
sigma = {el: nums[2 * i]/10 for i, el in enumerate(elements)}
epsilon = {el: nums[2 * i+1]*4.184 for i, el in enumerate(elements)}
numatoms = {'F':9, 'C':6, 'H':1, 'Cl':17, 'N':7, 'O':8, 'S':16, 'Br':35, 'I':53}

excel_data = pd.read_excel('/mnt/c/ПРоект/Edispandnamesx40x10(1).xlsx')
data = pd.DataFrame(excel_data, columns=['имя без цифр', 'mol1', 'mol2', 'E dissotiation', 'Full file name'])

names=[]
for file in os.listdir('/mnt/c/ПРоект/файлы/ELF3_num/'):
    names.append('/mnt/c/ПРоект/файлы/ELF3_num/'+file)
forcefield = app.ForceField(*names)


def OPLS_LJ(system, sigma, epsilon):
    forces = {system.getForce(index).__class__.__name__: system.getForce(
        index) for index in range(system.getNumForces())}
    nonbonded_force = forces['NonbondedForce']
    lorentz = CustomNonbondedForce(
        '4*epsilon*((sigma/r)^12); sigma=(sigma1+sigma2)/2; epsilon=sqrt(epsilon1*epsilon2)')
    lorentz.setNonbondedMethod(nonbonded_force.getNonbondedMethod())
    lorentz.addPerParticleParameter('sigma')
    lorentz.addPerParticleParameter('epsilon')
    lorentz.setCutoffDistance(nonbonded_force.getCutoffDistance())
    system.addForce(lorentz)
    LJset = {}
    for index in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
        LJset[index] = (sigma, epsilon)
        lorentz.addParticle([sigma, epsilon])
        nonbonded_force.setParticleParameters(
            index, charge, sigma, epsilon*0)
    for i in range(nonbonded_force.getNumExceptions()):
        (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
        # ALL THE 12,13 and 14 interactions are EXCLUDED FROM CUSTOM NONBONDED
        # FORCE
        lorentz.addExclusion(p1, p2)
        if eps._value != 0.0:
            #print p1,p2,sig,eps
            sig14 = (LJset[p1][0] + LJset[p2][0])/2
            eps14 = sqrt(abs(LJset[p1][1] * LJset[p2][1]))
            nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)
    return system

def getting_openmm_Energy(file, sigma, epsilon, forcefield):
    mol = PDBFile('/mnt/c/ПРоект/файлы/all_pdb/'+file)
    model = Modeller(mol.topology, mol.getPositions())
    model.addExtraParticles(forcefield)
    system = forcefield.createSystem(model.topology)
    positions = model.positions
    system = OPLS_LJ(system, sigma, epsilon)
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
    simulation = Simulation(model.topology, system, integrator)
    simulation.context.setPositions(model.positions)
    E = simulation.context.getState(getEnergy=True)
    OpenmmEnergy =E.getPotentialEnergy() / openmm.unit.kilocalorie_per_mole
    return OpenmmEnergy 

def pyD3Energy(file, ns6, ns8, na1, na2):
    with open('/mnt/c/ПРоект/файлы/all_pdb/'+file, 'r') as f:
        lines = f.readlines()
    atoms = lines[2:]
    coords = []
    elements = []
    for atom in atoms:
        data = atom.split()
        if data[0]=='HETATM':
            elements.append (data[10])
            coords.append ([float(float(d)/0.52917721090380) for d in data[5:8]]) #in Bohr
    numbers =np.array([numatoms[c] for c in elements])
    positions = np.array(coords)
    model = DispersionModel (numbers, positions)
    res = model.get_dispersion(RationalDampingParam(s6=ns6, s8=ns8, a1=na1, a2=na2), grad=False)
    EpyD3=(res.get("energy"))*((4.359744722207185e-18)*6.02e23*(1e-3)/4.198) # ккал/моль  #энергия хартри(википедия)
    return EpyD3

def Eopenmm_plus_ED3(file, sigma, epsilon, ns6, ns8, na1, na2, forcefield):
    return getting_openmm_Energy(file, sigma, epsilon, forcefield)+pyD3Energy(file, ns6, ns8, na1, na2)

file=input()  #'benBracetone095.pdb'
my_file = open("/mnt/c/ПРоект/вывод/new.txt", "a")
print(file[:-4], str(Eopenmm_plus_ED3(file, sigma, epsilon, ns6, ns8, na1, na2, forcefield)), file=my_file)
my_file.close()
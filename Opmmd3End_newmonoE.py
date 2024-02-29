import sys
from openmm.app import ForceField, PDBReporter, Element
from openmm import LangevinMiddleIntegrator, CustomNonbondedForce
from openmm.app import PDBFile, ForceField, Modeller, Simulation, Topology
from typing import Dict, List, Tuple
from openmm.unit import kilocalorie_per_mole, picosecond, picoseconds, kelvin, nanometer
import os
from openmm import app
import argparse
from FilesProcessing import parse_pdb, Molecule
from dftd3.interface import RationalDampingParam, DispersionModel, Structure
import numpy as np
from datetime import datetime
import pandas as pd
import glob
import shutil
import scipy
#4.51 {'C': 0.046258219623268815, 'H': 9.951413855720602} {'C': 0.04968332123826258, 'H': 0.07997391760000028} 0.046258219623268815 0.04968332123826258 9.951413855720602 0.07997391760000028
elements = ['C', 'H']
nums = [4000.6, 1000.0, 4000.6, 1000.0, 1.0, 0.9, 1.0, 5.4]
#2692.122906888303 {'F': 3.086410904684778, 'C': 0.44574631427014816, 'H': 0.20041186844494427, 'CL': 1.5074512895318457, 'N': 0.02075356967828492, 'O': 1.6995717243420019, 'S': 3.9132387191762588, 'BR': 3.509990658535857, 'I': 6.203888133870684} 
#{'F': 0.03400494789671138, 'C': 0.1315348366615912, 'H': 0.08586591352378872, 'CL': 1.665900068231326, 'N': 1.020839267515339, 'O': 0.10020419708307332, 'S': 1.6788908966530898, 'BR': 0.8144528813676861, 'I': 0.8769040526010099} 0.8303073829750773 0.7410360670718377 0.8034726086760405 3.9950696230627987
#nums = [12.0, 1507.0, 12.0, 1507.0, 12.0, 1507.0, 12.0, 1507.0, 12.0, 1507.0, 12.0, 1507.0, 12.0, 1507.0, 12.0, 1507.0, 12.0, 1507.0, 1.0, 0.5, 1.2, 5.0]
#ns6, ns8, na1, na2 = [1.0, 0.9, 0.3385, 2.88]
ns6, ns8, na1, na2 = nums[-4], nums[-3], nums[-2], nums[-1]
sigma = {el: nums[2 * i] for i, el in enumerate(elements)}
epsilon = {el: nums[2 * i+1] for i, el in enumerate(elements)}
numatoms = {'C':6, 'H':1}

xml_num="/home/xray/andreenko/optimization/xml_num/"
xml_word="/home/xray/andreenko/optimization/xml_word/"

NM_TO_BOHR=0.052917721090380

excel_data = pd.read_excel("/home/xray/andreenko/optimization/Ediss_all.xlsx")
data = pd.DataFrame(excel_data, columns=['name', 'E_diss'])
print(sigma)
print(epsilon)
names=[]
for file in os.listdir(xml_num):
    names.append(xml_num+file)
forcefield = app.ForceField(*names)


def to_openmm(mol: Molecule):
    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue('UNK', chain)
    elements = []
    atoms = []
    for atom in mol.atoms:
        atoms.append(topology.addAtom(
            atom.type, Element.getBySymbol(atom.element),
            residue, atom.num))
        elements.append(str(atom).split()[-1])
    #print(elements)
    for bond in mol.get_bonds():
        topology.addBond(
            atoms[mol.atoms.index(bond[0])],
            atoms[mol.atoms.index(bond[1])]
        )
    positions = [a.coords / 10 for a in mol.atoms]
    numbers =[numatoms[c] for c in elements]
    return topology, positions, numbers


def OPLS_LJ(system, sigma, epsilon):
    forces = {system.getForce(index).__class__.__name__: system.getForce(
        index) for index in range(system.getNumForces())}
    nonbonded_force = forces['NonbondedForce']
    lorentz = CustomNonbondedForce(
        'epsilon*(1/r)*exp(-sigma*((r)^2)); sigma=(sigma1+sigma2)/2; epsilon=sqrt(epsilon1*epsilon2)')
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
            eps14 = np.sqrt(abs(LJset[p1][1] * LJset[p2][1]))
            nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps14)
    return system


def calc_energy(sigma, epsilon, ns6, ns8, na1, na2, numrs, topology: Topology, positions: List[Tuple[float, float, float]], ff, name: str):
    
    model = Modeller(topology, positions)
    model.addExtraParticles(ff)
    sys = ff.createSystem(model.topology)
    sys = OPLS_LJ(sys, sigma, epsilon)
    integrator = LangevinMiddleIntegrator(
        300 * kelvin,
        1 / picosecond,
        0.004 * picoseconds)
    sim = Simulation(model.topology, sys, integrator)
    sim.context.setPositions(model.positions)
    #if not os.path.exists('reports'):
    #    os.makedirs('reports')
    #reporter = PDBReporter('reports/%s.pdb' % name, 1)
    #reporter.report(sim, sim.context.getState(getPositions=True))
    E = sim.context.getState(getEnergy=True)
    OpenmmEnergy =E.getPotentialEnergy() / kilocalorie_per_mole
    posit=np.array(positions)/NM_TO_BOHR
    numbers=np.array(numrs, dtype=int)
    mod = DispersionModel (numbers, posit)
    res = mod.get_dispersion(RationalDampingParam(s6=ns6, s8=ns8, a1=na1, a2=na2), grad=False)
    EpyD3=(res.get("energy"))*((4.359744722207185e-18)*6.02e23*(1e-3)/4.198) 
    return OpenmmEnergy+EpyD3

def E_diss_for_all(sigma, epsilon, ns6, ns8, na1, na2):
    
    pdbs = glob.glob("/home/xray/andreenko/optimization/opt_pdb/*")
    xmls = [os.path.join(xml_num, f) for f in os.listdir(xml_num)]
    ff = ForceField(*xmls)
    energies = {}
    Eerrors = {}
    for pdb in pdbs:
        #print(pdb)
        try:
            name, _ = os.path.splitext(os.path.basename(pdb))

            mol = PDBFile(pdb)
            _, components, _ = parse_pdb(pdb)
            topologies = []
            positions = []
            numrs=[]
            for component in components:
                #print(component)
                t, p, num = to_openmm(component)
                #print(t)
                ##rint(num)
                topologies.append(t)
                #print(3)
                positions.append(p)
                #print(5)
                numrs.append(num)
                #print(6)
            #print(len(numrs), type(numrs[0]))
            dim_posit=mol.getPositions()/nanometer
            dim_numrs=[a.element.atomic_number for a in mol.topology.atoms()]
            #print(dim_posit)
            #exit(0)
            e = calc_energy(sigma, epsilon, ns6, ns8, na1, na2, dim_numrs, mol.topology, dim_posit,
                                         ff, name)
            energies[name] = e
            diss_energy = e

            for i, (t, p, num) in enumerate(zip(topologies, positions, numrs)):
                component_name = '%s_mol%02d' % (name, i + 1)
                #print(component_name, num, numrs)
                e = calc_energy(sigma, epsilon, ns6, ns8, na1, na2, num, t, p, ff,
                                component_name)
                energies[component_name] = e
                diss_energy -= e
            energies['%s_diss' % name] = diss_energy

            filename=pdb.split('/')[-1][:-4]
            line = data[data['name'] == filename]
            Etable=float(line['E_diss'].loc[line.index[0]])
            Eerrors[name]= abs(diss_energy - Etable)
        except Exception as e:
            print('Error during calculation of %s. Error is %s' % (pdb, repr(e)), file=sys.stderr)
            exit(1)
    return energies, Eerrors
#print(E_diss_for_all(sigma, epsilon, ns6, ns8, na1, na2))

def change_sigma_n_epsilon(file, sigma, epsilon, ns6, ns8, na1, na2):
    filename = xml_word + file
    shutil.copyfile(filename, xml_num+file)
    #print(file)
    with open(xml_num+file, 'r') as f:
        t = f.read()
    for e in elements:
        #print(e, sigma.get(e))
        t = t.replace('{{%s_sigma}}'%e, '%f'%sigma.get(e))
        t = t.replace('{{%s_epsilon}}'%e, '%f'%epsilon.get(e))
    with open(xml_num+file, 'w+') as c:
        c.write(t)

def Esum_allfiles(nums):
    elements = ['C', 'H']
    #ns6, ns8, na1, na2 = [1.0, 0.9, 0.3385, 2.88]
    ns6, ns8, na1, na2 = nums[-4], nums[-3], nums[-2], nums[-1]
    sigma = {el: abs(nums[2 * i]) for i, el in enumerate(elements)}
    epsilon = {el: abs(nums[2 * i+1]) for i, el in enumerate(elements)}
    for file in os.listdir(xml_word):
        change_sigma_n_epsilon(file, sigma, epsilon, ns6, ns8, na1, na2)
    forcefield = app.ForceField(*names)
    energies, Eerrors = E_diss_for_all(sigma, epsilon, ns6, ns8, na1, na2)
    #print(energies, Eerrors)
    total_Ener = sum(Eerrors.values())
    current_datetime = datetime.now().strftime("%Y-%m-%d")
    str_current_datetime = str(current_datetime)
    file_name = "/home/xray/andreenko/optimization/optimize/"+'optimize_parametrs_tkachneopent54'+str_current_datetime+".txt"
    file = open(file_name, 'a')
    print(total_Ener, sigma, epsilon, ns6, ns8, na1, na2, file=file)
    file.close()
    return total_Ener

#result = scipy.optimize.minimize(Esum_allfiles, nums, method = 'Nelder-Mead', options={'disp': True})
result = scipy.optimize.dual_annealing(Esum_allfiles, bounds=[(0, 10000), (0, 50000), (0, 10000), (0, 50000)])
print(result.x)
#bounds=[(None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (0.2, 1.7), (0.0, 1.5), (-1.0, 5.0), (0.0, 10.0)]
#file='benBracetone095.pdb'
#print(Eopenmm_plus_ED3(file, sigma, epsilon, ns6, ns8, na1, na2, forcefield))

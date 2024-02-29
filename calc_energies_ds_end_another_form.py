import sys
from openmm.app import ForceField, PDBReporter, Element
from openmm import LangevinMiddleIntegrator, CustomNonbondedForce
from openmm.app import PDBFile, ForceField, Modeller, Simulation, Topology
from typing import Dict, List, Tuple
from openmm.unit import kilocalorie_per_mole, picosecond, picoseconds, kelvin, nanometer
import os
import shutil
import argparse
from FilesProcessing import parse_pdb, Molecule
from dftd3.interface import RationalDampingParam, DispersionModel, Structure
import numpy as np
#from /mnt/c/ПРоект/Github/Energy_calculation/onlyEnergy.py import *
import json

NM_TO_BOHR=0.052917721090380

#ns6, ns8, na1, na2 = 1.2997192618309348, 1.1315029169023156, 0.5239447802856281, 3.4059508470675146
#sigma = {'F': 0.26878280174014524, 'C': 0.33692970113076093, 'H': 0.1516303732010405, 'Cl': 0.3190836343734404, 'N': 0.23807485526025784, 'O': 0.22621607436939062, 'S': 0.30688554958413905, 'Br': 0.3097643362352015, 'I': 0.3526265339794298}
#epsilon = {'F': 0.2550915916812575, 'C': 0.2687946060168284, 'H': 0.11360354855351353, 'Cl': 1.114949928461474, 'N': 0.6711781962912372, 'O': 0.9929899117495553, 'S': 1.5028358632628214, 'Br': 2.7225027636889827, 'I': 1.7168250879301197}
#numatoms = {'F':9, 'C':6, 'H':1, 'CL':17, 'N':7, 'O':8, 'S':16, 'BR':35, 'I':53, 'AR':18}
#elements = ['F', 'C', 'H', 'CL', 'N', 'O', 'S', 'BR', 'I', 'AR']
elements=['C', 'H']
numatoms={'C':6, 'H':1}
#elements=['AR']
#numatoms={'AR':18}

def change_sigma_n_epsilon(xml_folder_with_words_instead, xml_folder_with_changed_values, sigma, epsilon, elements):
    for filename in os.listdir(xml_folder_with_words_instead):
        shutil.copyfile(xml_folder_with_words_instead+filename, xml_folder_with_changed_values+filename)
        with open(xml_folder_with_changed_values+filename, 'r') as f:
            t = f.read()
        for e in elements:
            #print(e)
            t = t.replace('{{%s_sigma}}'%e, '%f'%float(sigma.get(e)))
            t = t.replace('{{%s_epsilon}}'%e, '%f'%float(epsilon.get(e)))
        with open(xml_folder_with_changed_values+filename, 'w+') as c:
            c.write(t)

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
        '(1/r)*epsilon*exp(-sigma*(r^2)); sigma=(sigma1+sigma2)/2; epsilon=sqrt(epsilon1*epsilon2)')
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


def calc_energy(sigma, epsilon, ns6, ns8, na1, na2, numrs, topology: Topology, positions: List[Tuple[float, float, float]],
                ff, name: str):
    
    model = Modeller(topology, positions)
    #print(4897)
    model.addExtraParticles(ff)
    #print(587)
    sys = ff.createSystem(model.topology)
    sys = OPLS_LJ(sys, sigma, epsilon)
    integrator = LangevinMiddleIntegrator(
        300 * kelvin,
        1 / picosecond,
        0.004 * picoseconds
    )
    #print(15)
    sim = Simulation(model.topology, sys, integrator)
    sim.context.setPositions(model.positions)
    #if not os.path.exists('reports'):
    #    os.makedirs('reports')
    #reporter = PDBReporter('reports/%s.pdb' % name, 1)
    #reporter.report(sim, sim.context.getState(getPositions=True))
    #print(20)
    E = sim.context.getState(getEnergy=True)
    OpenmmEnergy =E.getPotentialEnergy() / kilocalorie_per_mole
    posit=np.array(positions)/NM_TO_BOHR
    numbers=np.array(numrs, dtype=int)
    mod = DispersionModel (numbers, posit)
    res = mod.get_dispersion(RationalDampingParam(s6=ns6, s8=ns8, a1=na1, a2=na2), grad=False)
    EpyD3=(res.get("energy"))*((4.359744722207185e-18)*6.02e23*(1e-3)/4.198) # ккал/моль  #энергия хартри(википедия)
    
    return OpenmmEnergy, EpyD3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('params', help='json file with params')
    parser.add_argument('xml_dir_w', help='directory with xml files with words instead')
    parser.add_argument('xml_dir_n', help='directory with xml files with changed values')
    parser.add_argument('out', help='name of the output file. .csv will be added by default')
    parser.add_argument('pdbs', nargs='+', help='space-separated list of .pdb files')
    args = parser.parse_args()
    
    with open(args.params, 'r') as json_file:
	    json_load = json.load(json_file)
    sigma = json_load["sigma"]
    epsilon = json_load["epsilon"]
    ns6, ns8, na1, na2 = json_load["ns6"], json_load["ns8"], json_load["na1"], json_load["na2"]

    change_sigma_n_epsilon(args.xml_dir_w, args.xml_dir_n, sigma, epsilon, elements)
    pdbs = args.pdbs
    xmls = [os.path.join(args.xml_dir_n, f) for f in os.listdir(args.xml_dir_n)]
    ff = ForceField(*xmls)
    energies = {}
    for pdb in pdbs:
        print(pdb)
        try:
            name, _ = os.path.splitext(os.path.basename(pdb))
            
            mol = PDBFile(pdb)
            #print(1)
            _, components, _ = parse_pdb(pdb)
            #print(components)
            topologies = []
            positions = []
            numrs=[]
            #print(2)
            for component in components:
                #print(component)
                t, p, num = to_openmm(component)
                #print(t)
                #print(num)
                topologies.append(t)
                #print(3)
                positions.append(p)
                #print(5)
                numrs.append(num)
                #print(6)
            #print(len(numrs), type(numrs[0]))
            dim_posit=mol.getPositions()/nanometer
            dim_numrs=[a.element.atomic_number for a in mol.topology.atoms()]
            #print('precalc')
            Opmm, pyD3 = calc_energy(sigma, epsilon, ns6, ns8, na1, na2, dim_numrs, mol.topology, dim_posit,
                                         ff, name)
            energies[name+'_Opmm'] = Opmm
            energies[name+'_pyD3'] = pyD3
            diss_energy = Opmm+pyD3

            for i, (t, p, num) in enumerate(zip(topologies, positions, numrs)):
                component_name = '%s_mol%02d' % (name, i + 1)
                #print(component_name, num, numrs)
                Opmm, pyD3 = calc_energy(sigma, epsilon, ns6, ns8, na1, na2, num, t, p, ff,
                                component_name)
                energies[component_name+'_Opmm'] = Opmm
                energies[component_name+'_pyD3'] = pyD3
                diss_energy -= Opmm+pyD3
            energies['%s_diss' % name] = diss_energy
        except Exception as e:
            raise e
            print('Error during calculation of %s. Error is %s' % (pdb, repr(e)), file=sys.stderr)
            exit(1)
    
    out = args.out
    if not out.endswith('.csv'):
        out += '.csv'
    output = []
    for name, energy in energies.items():
        output.append('%s,%f\n' % (name, energy))
    with open(out, 'w+') as o:
        o.writelines(output)

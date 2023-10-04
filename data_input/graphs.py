import os
import numpy as np
import pymatgen.core as mg
import json
import pickle
from pymatgen.core import Structure
from megnet.data.crystal import CrystalGraph
import warnings
import time
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore")

C_FORBIDDEN_ELEMENTS = ["He", "Ne", "Ar", "Kr", "Xe"]
C_FORBIDDEN_ELEMENTS_Z = [2, 10, 18, 36, 54]
N_pool = 8 # This is our cap on multiprocessing
K = 48
C_SUPERCELL = (1,1,1)

def convert_structure_to_reciprocal(structure):
	"""
	Takes in a crystal structure and returns the structure
	with site positions relative to the reciprocal lattice.

	Args:
		structure : pymatgen.core.Structure
			The crystal structure to convert.
	Returns:
		pymatgen.core.Structure
		Original crystal structure with lattice converted to that of the original's reciprocal space.
	"""
	reciprocal_lattice_matrix	= structure.lattice.reciprocal_lattice.matrix
	return mg.Structure(reciprocal_lattice_matrix, structure.species, structure.frac_coords)

def combine_real_reciprocal_graphs(g_real, g_reciprocal):
	"""
	Takes the real and reciprocal graphs of a crystal structure
	and combines them into one single multiplex graph.

	Args:
		g_real : dict
			Graph of the structure in real space
		g_reciprocal : dict
			Graph of the structure in reciprocal space
	Returns:
		dict
		Single combined multiplex graph 
	"""
	out_dict = {}
	out_dict["atom"] = g_real["atom"]
	out_dict["real_bond"]		= g_real["bond"]
	out_dict["real_state"]		= g_real["state"]
	out_dict["real_index1"]		= g_real["index1"]
	out_dict["real_index2"]		= g_real["index2"]
	out_dict["reciprocal_bond"]		= g_reciprocal["bond"]
	out_dict["reciprocal_state"]	= g_reciprocal["state"]
	out_dict["reciprocal_index1"]	= g_reciprocal["index1"]
	out_dict["reciprocal_index2"]	= g_reciprocal["index2"]
	out_dict["combined_state"] = np.array([[0.,0.]],dtype="float32")
	return out_dict

def graph_has_isolated_atoms(graph):
	"""
	Checks if graph has isolated atoms with no connections.

	Args:
		graph: dict
			Crystal structure graph

	Returns:
		bool
			Returns True if the graph contains an isolated atom, and False otherwise.
	"""
	# The atoms are labeled from 0...(N-1) for N atoms
	# Because all bonds are bi-directional, if a number is missing from index1 it is isolated
	if np.size(np.unique(graph["index1"])) < len(graph["atom"]):
		return True
	return False

def structure_contains_forbidden_elements(structure):
	"""
	Checks if structure contains any of the global forbidden elements defined by C_FORBIDDEN_ELEMENTS.

	Args:
		graph: dict
			Crystal structure graph

	Returns:
		bool
			Returns True if the graph contains an isolated atom, and False otherwise.
	"""

	species = structure.species
	for specie in species:
		if specie.symbol in C_FORBIDDEN_ELEMENTS:
			return True
	return False

def structure_to_dual_graph(t_structure, real_crystal_graph_generator, reciprocal_graph_generator, graph_cull = False, cull_len = 10000):
	"""
	Takes in a crystal structure and creates a full multiplex graph, and can cull overly large graphs.
	Args:
		t_structure: pymatgen.core.Structure
			Input crystal structure 
		real_crystal_graph_generator: megnet.data.crystal.CrystalGraph
			Megnet crystal graph generator for real space structure
		reciprocal_graph_generator: megnet.data.crystal.CrystalGraph
			Megnet crystal graph generator for reciprocal space structure
		graph_cull: bool
			Turn on/off culling of large graphs. If they're large, will return None
		cull_len: int
			Maximum number of edges the graph can contain before being culled

	Returns:
		dict
			Returns combined real and reciprocal multiplex graph of crystal structure.
	"""
	t_structure_reciprocal = convert_structure_to_reciprocal(t_structure)
	t_graph = real_crystal_graph_generator.convert(t_structure)
	# Isolated check
	if graph_has_isolated_atoms(t_graph):
		return None
	t_graph_reciprocal = reciprocal_graph_generator.convert(t_structure_reciprocal)
	if graph_has_isolated_atoms(t_graph_reciprocal):
		return None
	t_combined_graph = combine_real_reciprocal_graphs(t_graph,t_graph_reciprocal)
	if graph_cull:
		if (t_combined_graph["reciprocal_index1"].shape[0] > cull_len):
			return None
	return t_combined_graph

def structure_to_dual_graph_KNN(t_structure, real_crystal_graph_generator, reciprocal_graph_generator, max_edges_real = 24, max_edges_reciprocal = 48, bHardFail = False):
	"""
	Takes in a crystal structure and creates a full multiplex graph, with edges trimmed to K-Nearest neighbours.
	Args:
		t_structure: pymatgen.core.Structure
			Input crystal structure 
		real_crystal_graph_generator: megnet.data.crystal.CrystalGraph
			Megnet crystal graph generator for real space structure
		reciprocal_graph_generator: megnet.data.crystal.CrystalGraph
			Megnet crystal graph generator for reciprocal space structure
		max_edges_real: int
			Sets the number of outgoing real-space edges for each node.
		max_edges_reciprocal: int
			Sets the number of outgoing reciprocal-space edges for each node.
		bHardFail: bool
			If an insufficient number of edges are found in either space for any node, it instead returns None.

	Returns:
		dict
			Returns combined real and reciprocal multiplex graph of crystal structure, with KNN edges.
	"""
	t_dual_graph = structure_to_dual_graph(t_structure, real_crystal_graph_generator, reciprocal_graph_generator, graph_cull = False)

	# Shortcut if failed
	if t_dual_graph is None:
		print("Structure to dual graph failed! Returned None.")
		return None


	# So iterate through each atom index
	atom_indices = np.unique(t_dual_graph["reciprocal_index1"])

	# Real
	out_bond_lengths_real = []
	out_index1s_real = []
	out_index2s_real = []

	# Reciprocal
	out_bond_lengths_reci = []
	out_index1s_reci = []
	out_index2s_reci = []

	# Iterate through each site's bonds
	bRealCutOk = True
	bReciCutOk = True
	for atom_idx in atom_indices:

		# For real first
		bond_arg_indices_real = np.where(t_dual_graph["real_index1"]==atom_idx)

		if len(bond_arg_indices_real[0]) < max_edges_real:
			bRealCutOk = False
			if bHardFail:
				print("Real cut radius too small! Graph will have under-quantity edges.")
				return None

		# Grab bonds and indices that are for the particular atom_idx
		idx_bond_lengths_real = t_dual_graph["real_bond"][bond_arg_indices_real]
		idx_bond_idx1_real    = t_dual_graph["real_index1"][bond_arg_indices_real]
		idx_bond_idx2_real    = t_dual_graph["real_index2"][bond_arg_indices_real]

		# Need to argsort the lengths and use the first N as the 
		bond_length_args_sorted_real = np.argsort(idx_bond_lengths_real)

		new_bond_lengths_real	= idx_bond_lengths_real[bond_length_args_sorted_real[:max_edges_real]]
		new_index1_real			= idx_bond_idx1_real[bond_length_args_sorted_real[:max_edges_real]]
		new_index2_real			= idx_bond_idx2_real[bond_length_args_sorted_real[:max_edges_real]]

		out_bond_lengths_real.append(new_bond_lengths_real)
		out_index1s_real.append(new_index1_real)
		out_index2s_real.append(new_index2_real)

		# Grab reciprocal bonds for atom index1
		bond_arg_indices_reci = np.where(t_dual_graph["reciprocal_index1"]==atom_idx)
		if len(bond_arg_indices_reci[0]) < max_edges_reciprocal:
			bReciCutOk = False
			if bHardFail:
				print("Reciprocal cut radius too small! Graph will have under-quantity edges.")
				return None

		idx_bond_lengths_reci = t_dual_graph["reciprocal_bond"][bond_arg_indices_reci]
		idx_bond_idx1_reci    = t_dual_graph["reciprocal_index1"][bond_arg_indices_reci]
		idx_bond_idx2_reci    = t_dual_graph["reciprocal_index2"][bond_arg_indices_reci]

		# Need to argsort the lengths and use the first N as the 
		bond_length_args_sorted = np.argsort(idx_bond_lengths_reci)

		new_bond_lengths = idx_bond_lengths_reci[bond_length_args_sorted[:max_edges_reciprocal]]
		new_index1 = idx_bond_idx1_reci[bond_length_args_sorted[:max_edges_reciprocal]]
		new_index2 = idx_bond_idx2_reci[bond_length_args_sorted[:max_edges_reciprocal]]

		out_bond_lengths_reci.append(new_bond_lengths)
		out_index1s_reci.append(new_index1)
		out_index2s_reci.append(new_index2)

	if not bRealCutOk:
		print("Real cut radius too small! Graph will have under-quantity edges.")
	if not bRealCutOk:
		print("Reciprocal cut radius too small! Graph will have under-quantity edges.")
	# Hstack them together
	out_bond_lengths_real	= np.hstack(out_bond_lengths_real)
	out_index1s_real		= np.hstack(out_index1s_real)
	out_index2s_real		= np.hstack(out_index2s_real)
	out_bond_lengths_reci	= np.hstack(out_bond_lengths_reci)
	out_index1s_reci		= np.hstack(out_index1s_reci)
	out_index2s_reci		= np.hstack(out_index2s_reci)

	# Rebuild the graph
	out_dict = {}
	out_dict["atom"]				= t_dual_graph["atom"]
	out_dict["real_bond"]			= out_bond_lengths_real
	out_dict["real_state"]			= t_dual_graph["real_state"]
	out_dict["real_index1"]			= out_index1s_real
	out_dict["real_index2"]			= out_index2s_real
	out_dict["reciprocal_bond"]		= out_bond_lengths_reci
	out_dict["reciprocal_state"]	= t_dual_graph["reciprocal_state"]
	out_dict["reciprocal_index1"]	= out_index1s_reci
	out_dict["reciprocal_index2"]	= out_index2s_reci
	out_dict["combined_state"]		= np.array([[0.,0.]],dtype="float32")

	return out_dict

def structure_to_dual_graph_KND(t_structure, real_crystal_graph_generator, reciprocal_graph_generator, max_edges_real = 24, max_edges_reciprocal = 48, bHardFail = False):
	"""
	Takes in a crystal structure and creates a full multiplex graph, with edges trimmed to K-Nearest distance (i.e. multiple edges at the same distance will be permitted, 
	presenting a consistent method of edge construction that preserves symmetry.)
	Args:
		t_structure: pymatgen.core.Structure
			Input crystal structure 
		real_crystal_graph_generator: megnet.data.crystal.CrystalGraph
			Megnet crystal graph generator for real space structure
		reciprocal_graph_generator: megnet.data.crystal.CrystalGraph
			Megnet crystal graph generator for reciprocal space structure
		max_edges_real: int
			Sets the number of outgoing real-space edges for each node.
		max_edges_reciprocal: int
			Sets the number of outgoing reciprocal-space edges for each node.
		bHardFail: bool
			If an insufficient number of edges are found in either space for any node, it instead returns None.

	Returns:
		dict
			Returns combined real and reciprocal multiplex graph of crystal structure, with KND edges.
	"""
	t_dual_graph = structure_to_dual_graph(t_structure, real_crystal_graph_generator, reciprocal_graph_generator, graph_cull = False)

	# Shortcut if failed
	if t_dual_graph is None:
		print("Structure to dual graph failed! Returned None.")
		return None


	# So iterate through each atom index
	atom_indices = np.unique(t_dual_graph["reciprocal_index1"])

	# Real
	out_bond_lengths_real = []
	out_index1s_real = []
	out_index2s_real = []

	# Reciprocal
	out_bond_lengths_reci = []
	out_index1s_reci = []
	out_index2s_reci = []

	# Iterate through each site's bonds
	bRealCutOk = True
	bReciCutOk = True
	for atom_idx in atom_indices:

		# For real first
		bond_arg_indices_real = np.where(t_dual_graph["real_index1"]==atom_idx)

		if len(bond_arg_indices_real[0]) < max_edges_real:
			bRealCutOk = False
			if bHardFail:
				print("Real cut radius too small! Graph will have under-quantity edges.")
				return None

		# Grab bonds and indices that are for the particular atom_idx
		idx_bond_lengths_real = t_dual_graph["real_bond"][bond_arg_indices_real]
		idx_bond_idx1_real    = t_dual_graph["real_index1"][bond_arg_indices_real]
		idx_bond_idx2_real    = t_dual_graph["real_index2"][bond_arg_indices_real]

		# Need to argsort the lengths and use the first N and get its distance 
		bond_length_args_sorted_real = np.argsort(idx_bond_lengths_real)

		# Need to determine updated max_edges_real
		idx_bond_lengths_real_sorted = idx_bond_lengths_real[bond_length_args_sorted_real]
		max_edges_real_cap = np.min([len(idx_bond_lengths_real_sorted),max_edges_real])
		max_D_real = idx_bond_lengths_real_sorted[max_edges_real_cap-1]
		max_K_real = len(idx_bond_lengths_real_sorted[idx_bond_lengths_real_sorted<=max_D_real])

		new_bond_lengths_real	= idx_bond_lengths_real[bond_length_args_sorted_real[:max_K_real]]
		new_index1_real			= idx_bond_idx1_real[bond_length_args_sorted_real[:max_K_real]]
		new_index2_real			= idx_bond_idx2_real[bond_length_args_sorted_real[:max_K_real]]

		out_bond_lengths_real.append(new_bond_lengths_real)
		out_index1s_real.append(new_index1_real)
		out_index2s_real.append(new_index2_real)

		# Grab reciprocal bonds for atom index1
		bond_arg_indices_reci = np.where(t_dual_graph["reciprocal_index1"]==atom_idx)
		if len(bond_arg_indices_reci[0]) < max_edges_reciprocal:
			bReciCutOk = False
			if bHardFail:
				print("Reciprocal cut radius too small! Graph will have under-quantity edges.")
				return None

		idx_bond_lengths_reci = t_dual_graph["reciprocal_bond"][bond_arg_indices_reci]
		idx_bond_idx1_reci    = t_dual_graph["reciprocal_index1"][bond_arg_indices_reci]
		idx_bond_idx2_reci    = t_dual_graph["reciprocal_index2"][bond_arg_indices_reci]

		# Need to argsort the lengths and use the first N and get its distance
		bond_length_args_sorted = np.argsort(idx_bond_lengths_reci)

		# Need to determine updated max_edges_real
		idx_bond_lengths_reci_sorted = idx_bond_lengths_reci[bond_length_args_sorted]
		max_edges_reci_cap = np.min([len(idx_bond_lengths_reci_sorted),max_edges_reciprocal])
		max_D_reci = idx_bond_lengths_reci_sorted[max_edges_reci_cap-1]
		max_K_reci = len(idx_bond_lengths_reci_sorted[idx_bond_lengths_reci_sorted<=max_D_reci])

		new_bond_lengths = idx_bond_lengths_reci[bond_length_args_sorted[:max_K_reci]]
		new_index1 = idx_bond_idx1_reci[bond_length_args_sorted[:max_K_reci]]
		new_index2 = idx_bond_idx2_reci[bond_length_args_sorted[:max_K_reci]]

		out_bond_lengths_reci.append(new_bond_lengths)
		out_index1s_reci.append(new_index1)
		out_index2s_reci.append(new_index2)

	if not bRealCutOk:
		print("Real cut radius too small! Graph will have under-quantity edges.")
	if not bRealCutOk:
		print("Reciprocal cut radius too small! Graph will have under-quantity edges.")
	# Hstack them together
	out_bond_lengths_real	= np.hstack(out_bond_lengths_real)
	out_index1s_real		= np.hstack(out_index1s_real)
	out_index2s_real		= np.hstack(out_index2s_real)
	out_bond_lengths_reci	= np.hstack(out_bond_lengths_reci)
	out_index1s_reci		= np.hstack(out_index1s_reci)
	out_index2s_reci		= np.hstack(out_index2s_reci)

	# Rebuild the graph
	out_dict = {}
	out_dict["atom"]				= t_dual_graph["atom"]
	out_dict["real_bond"]			= out_bond_lengths_real
	out_dict["real_state"]			= t_dual_graph["real_state"]
	out_dict["real_index1"]			= out_index1s_real
	out_dict["real_index2"]			= out_index2s_real
	out_dict["reciprocal_bond"]		= out_bond_lengths_reci
	out_dict["reciprocal_state"]	= t_dual_graph["reciprocal_state"]
	out_dict["reciprocal_index1"]	= out_index1s_reci
	out_dict["reciprocal_index2"]	= out_index2s_reci
	out_dict["combined_state"]		= np.array([[0.,0.]],dtype="float32")

	return out_dict

def data_json_to_KNN_dual_graphs(data_json, real_crystal_graph_generator, reciprocal_graph_generator, max_edges_real = 24, max_edges_reciprocal = 48, bHardFail = True, bUseCheckpoints = False, checkpoint_path = None, supercell = (1,1,1)):
	"""
	Takes in a dataset json of type provided in the megnet paper of MP2018/9 datasets, and for each structure creates a full multiplex graph, with edges trimmed to K-Nearest Neighbours. Can dump in periodic checkpoints of 5000 graphs.
	Args:
		data_json: dict
			Json contents from the MP2018/9 datasets provided by MEGNET paper.
		real_crystal_graph_generator: megnet.data.crystal.CrystalGraph
			Megnet crystal graph generator for real space structure
		reciprocal_graph_generator: megnet.data.crystal.CrystalGraph
			Megnet crystal graph generator for reciprocal space structure
		max_edges_real: int
			Sets the number of outgoing real-space edges for each node.
		max_edges_reciprocal: int
			Sets the number of outgoing reciprocal-space edges for each node.
		bHardFail: bool
			If an insufficient number of edges are found in either space for any node, it instead returns None.
		bUseCheckpoints: bool
			Will periodically dump checkpoints of 5000 structures to file during the process to prevent catastrophic loss of data
		checkpoint_path: str
			If using checkpoints, the directory to which the checkpoints will be saved.
		supercell: tuple[int]
			Tuple determining the number of images along (a,b,c) directions to produce graphs of supercells.

	Returns:
		tuple: list, np.array, list
			Returns a tuple of the graphs, associated property targets, and a list of any indices of structures that failed.
	"""
	
	if bUseCheckpoints and checkpoint_path is None:
		print("Trying to use checkpoints without setting path. FAILED.") # TODO fix better than this
		return
	graphs_out = []
	failed_index = []
	t_ys = []
	for i,datapoint in enumerate(data_json):
		if i % 5000==0 and bUseCheckpoints:
			root_folder = checkpoint_path
			with open(root_folder+f"checkpoint.pkl",'wb') as f:
				pickle.dump({"X":graphs_out,"y":t_ys,"failed_idxs":failed_index},f)
		try:
			t_structure_str = datapoint["structure"]
			t_y = datapoint["formation_energy_per_atom"]
			t_structure = Structure.from_str(t_structure_str,fmt="cif")*supercell
			if structure_contains_forbidden_elements(t_structure):
				failed_index.append(i)
				continue
			t_graph = structure_to_dual_graph_KNN(t_structure, real_crystal_graph_generator, reciprocal_graph_generator, max_edges_real, max_edges_reciprocal, bHardFail)
			if not t_graph is None:
				graphs_out.append(t_graph)
				t_ys.append(t_y)
			else:
				failed_index.append(i)
		except Exception as E:
			print(E)
			failed_index.append(i)
		if i%100 == 0:
			print(f"Progress: {i} / {len(data_json)}")
		
	if bUseCheckpoints:
		with open(root_folder+f"dataset_complete.pkl",'wb') as f:
					pickle.dump({"X":graphs_out,"y":t_ys,"failed_idxs":failed_index},f)
	return graphs_out, t_ys, failed_index

def get_cutoff_from_structure(structure):
	"""
	Calculates an approximate radial cutoff to use to prevent explosion of edges
	when constructing real space crystal graphs at fixed-distance, assisting in creation
	of KNN/KND structures.

	Args:
		structure : pymatgen.core.Structure
			The crystal structure to calculate real-space cutoff distance to attain

	Returns:
		float
			Returns suggested radial cutoff.
	"""
	atoms_per_vol = len(structure.sites) / structure.volume
	return 1.3*(3*K/(4*np.pi*atoms_per_vol))**(1/3) + K/50

def get_reciprocal_cutoff_from_structure(reci_structure):
	"""
	Calculates an approximate radial cutoff to use to prevent explosion of edges
	when constructing reciprocal space crystal graphs at fixed-distance, assisting in creation
	of KNN/KND structures.

	Args:
		structure : pymatgen.core.Structure
			The crystal structure to calculate reciprocal-space cutoff distance to attain

	Returns:
		float
			Returns suggested radial cutoff.
	"""
	atoms_per_vol = len(reci_structure.sites) / reci_structure.volume
	return 1.1*(3*K/(4*np.pi*atoms_per_vol))**(1/3)

def mp_s2dgknn(structure_str):
	"""
	Takes in a cif structure, and converts it to a K-nearest neighbours multiplex graph, using
	default parameters found to work well, suitable for multiprocessing.

	Args:
		structure_str : str
			The crystal structure in cif string format

	Returns:
		dict
			Multiplex graph dict if successful, or None if failed. 
	"""

	structure = Structure.from_str(structure_str,fmt="cif")
	if structure_contains_forbidden_elements(structure):
		return None
	structure = structure*C_SUPERCELL
	kstructure = convert_structure_to_reciprocal(structure)
	real_cutoff = get_cutoff_from_structure(structure)
	reciprocal_cutoff = get_reciprocal_cutoff_from_structure(kstructure)
	sg_real = CrystalGraph(cutoff=real_cutoff)
	sg_reci = CrystalGraph(cutoff = reciprocal_cutoff)
	try: 
		t_graph = structure_to_dual_graph_KNN(structure,sg_real,sg_reci,K,K,False)
		return t_graph
	except Exception as E:
		print(E)
		return None

def mp_s2dgknd(structure_str):
	"""
	Takes in a cif structure, and converts it to a K-nearest distance multiplex graph, using
	default parameters found to work well, suitable for multiprocessing.

	Args:
		structure_str : str
			The crystal structure in cif string format

	Returns:
		dict
			Multiplex graph dict if successful, or None if failed. 
	"""
	structure = Structure.from_str(structure_str,fmt="cif")
	if structure_contains_forbidden_elements(structure):
		return None
	structure = structure*C_SUPERCELL
	kstructure = convert_structure_to_reciprocal(structure)
	real_cutoff = get_cutoff_from_structure(structure)
	reciprocal_cutoff = get_reciprocal_cutoff_from_structure(kstructure)
	sg_real = CrystalGraph(cutoff=real_cutoff)
	sg_reci = CrystalGraph(cutoff = reciprocal_cutoff)
	try: 
		t_graph = structure_to_dual_graph_KND(structure,sg_real,sg_reci,K,K,False)
		return t_graph
	except Exception as E:
		print(E)
		return None

def flatten_dual_dataset(tX):
	"""
	"Flattens" a dual-graph dataset to enable it for use with dual_batch_generator.

	Args:
		tX : list[Dict]
			Input crystal structure dual graphs
	Returns:
		list[list]
			The graphs "flattened" into corresponding list on a per-part basis.
	"""
	x0 = [_["atom"] for _ in tX]
	x1 = [_["real_bond"] for _ in tX]
	x2 = [_["real_state"] for _ in tX]
	x3 = [_["real_index1"] for _ in tX]
	x4 = [_["real_index2"] for _ in tX]
	x5 = [_["reciprocal_bond"] for _ in tX]
	x6 = [_["reciprocal_state"] for _ in tX]
	x7 = [_["reciprocal_index1"] for _ in tX]
	x8 = [_["reciprocal_index2"] for _ in tX]
	return [x0,x1,x2,x3,x4,x5,x6,x7,x8]

def flatten_dataset(tX):
	"""
	"Flattens" a single-graph dataset to enable it for use with batch_generator.

	Args:
		tX : list[Dict]
			Input crystal structure dual graphs
	Returns:
		list[list]
			The graphs "flattened" into corresponding list on a per-part basis.
	"""		
	x0 = [_["atom"] for _ in tX]
	x1 = [_["real_bond"] for _ in tX]
	x2 = [_["real_state"] for _ in tX]
	x3 = [_["real_index1"] for _ in tX]
	x4 = [_["real_index2"] for _ in tX]
	return [x0,x1,x2,x3,x4]
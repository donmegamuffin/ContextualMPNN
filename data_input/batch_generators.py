from abc import abstractmethod
from inspect import signature
from operator import itemgetter
from logging import getLogger
from typing import Union, Dict, List, Any

import numpy as np
from monty.json import MSONable
from tensorflow.keras.utils import Sequence
from pymatgen.core import Structure
from pymatgen.analysis.local_env import NearNeighbors

from megnet.data import local_env
from megnet.utils.data import get_graphs_within_cutoff
from megnet.utils.general import expand_1st, to_list
from operator import itemgetter

from megnet.data.graph import BaseGraphBatchGenerator

def itemgetter_list(data_list: List, indices: List) -> tuple:
    """
    Get indices of data_list and return a tuple
    Args:
        data_list (list):  data list
        indices: (list) indices
    Returns:
        tuple
    """
    it = itemgetter(*indices)
    if np.size(indices) == 1:
        return (it(data_list),)
    return it(data_list)

class DualGraphBatchGenerator(BaseGraphBatchGenerator):
	"""
	A generator class that generates N pairs of multiplex graphs
	for input into standard MEGNET-style framework.
	"""

	def __init__(
		self,
		atom_features: List[np.array],
		bond_features_real: List[np.array],
		state_features_real: List[np.array],
		index1_list_real: List[int],
		index2_list_real: List[int],
		bond_features_reciprocal: List[np.array],
		state_features_reciprocal: List[np.array],
		index1_list_reciprocal: List[int],
		index2_list_reciprocal: List[int],
		targets: np.array = None,
		sample_weights: np.array = None,
		batch_size: int = 128,
		is_shuffle: bool = True,
	):
		"""
		Args:
			atom_features: list[np.array]
				list of atom feature matrix
			bond_features_real: list[np.array]
				list of bond real features matrix
			state_features_real: list[np.array] 
				list of [1, G] real state features, where G is the global state feature dimension
			index1_list_real: list[int] 
				list of (M, ) one side atomic index of the real bond,
				M is different for different structures
			index2_list_real: list[int]  
				list of (M, ) the other side atomic	index of the real bond, M is different for different structures,
				but it has to be the same as the corresponding x3_temp.
			bond_features_reciprocal: list[np.array]
				list of reciprocal bond features matrix
			state_features_reciprocal: list[np.array] 
				list of [1, G] reciprocal state features, where G is the global state feature dimension
			index1_list_reciprocal: list[int] 
				list of (M, ) one side atomic index of the reciprocal bond,
				M is different for different structures
			index2_list_reciprocal: list[int]  
				list of (M, ) the other side atomic	index of the reciprocal bond, M is different for different structures,
				but it has to be the same as the corresponding x3_temp.
			targets: np.array, 
				N*1, where N is the number of structures
			sample_weights: (numpy array), 
				N*1, where N is the number of structures
			batch_size: (int) 
				number of samples in a batch
		"""
		super().__init__(
			len(atom_features), targets, sample_weights=sample_weights, batch_size=batch_size, is_shuffle=is_shuffle
		)
		self.atom_features = atom_features
		self.bond_features_real = bond_features_real
		self.state_features_real = state_features_real
		self.index1_list_real = index1_list_real
		self.index2_list_real = index2_list_real
		self.bond_features_reciprocal = bond_features_reciprocal
		self.state_features_reciprocal = state_features_reciprocal
		self.index1_list_reciprocal = index1_list_reciprocal
		self.index2_list_reciprocal = index2_list_reciprocal

	def _generate_inputs(self, batch_index: list) -> tuple:
		"""Get the graph descriptions for each batch
		Args:
			 batch_index: List[int] 
				List of indices for training batch
		Returns:
			(tuple): Input arrays describe each network:
				np.array : List of features for each nodes
				np.array : List of real features for each edge
				np.array : List of real global state for each graph
				np.array : List of real indices for the start of each edge
				np.array : List of real indices for the end of each edge
				np.array : List of reciprocal features for each edge
				np.array : List of reciprocal global state for each graph
				np.array : List of reciprocal indices for the start of each edge
				np.array : List of reciprocal indices for the end of each edge
		"""

		# Get the features and connectivity lists for this batch
		x0_temp = itemgetter_list(self.atom_features, batch_index)
		x1_temp = itemgetter_list(self.bond_features_real, batch_index)
		x2_temp = itemgetter_list(self.state_features_real, batch_index)
		x3_temp = itemgetter_list(self.index1_list_real, batch_index)
		x4_temp = itemgetter_list(self.index2_list_real, batch_index)
		x5_temp = itemgetter_list(self.bond_features_reciprocal, batch_index)
		x6_temp = itemgetter_list(self.state_features_reciprocal, batch_index)
		x7_temp = itemgetter_list(self.index1_list_reciprocal, batch_index)
		x8_temp = itemgetter_list(self.index2_list_reciprocal, batch_index)

		return x0_temp, x1_temp, x2_temp, x3_temp, x4_temp, x5_temp, x6_temp, x7_temp, x8_temp

	def _combine_graph_data(
		self,
		x0_temp: List[np.array],	# Node feat
		x1_temp: List[np.array],	# Real Edge Feat
		x2_temp: List[np.array],	# Real State
		x3_temp: List[np.array],	# Real x3_temp
		x4_temp: List[np.array],	# Real x4_temp
		x5_temp: List[np.array],	# Reciprocal Edge Feat
		x6_temp: List[np.array],	# Reciprocal State
		x7_temp: List[np.array],	# Reciprocal x3_temp
		x8_temp: List[np.array],	# Reciprocal x4_temp
	) -> tuple:
		"""Compile the matrices describing each graph into single matrices for the entire graph
		Beyond concatenating the graph descriptions, this operation updates the indices of each
		node to be sequential across all graphs so they are not duplicated between graphs
		Args:
			x0_temp np.array: List of features for each nodes
			x1_temp np.array: List of real features for each edge
			x2_temp np.array: List of real global state for each graph
			x3_temp np.array: List of real indices for the start of each edge
			x4_temp np.array: List of real indices for the end of each edge
			x5_temp np.array: List of reciprocal features for each edge
			x6_temp np.array: List of reciprocal global state for each graph
			x7_temp np.array: List of reciprocal indices for the start of each edge
			x8_temp np.array: List of reciprocal indices for the end of each edge
		Returns:
			(tuple): Input arrays for the model describing the entire batch of graphs:
				- np.array : Features for each node
				- np.array : Features for each real edge
				- np.array : Global state for each real state
				- np.array : Indices for the start of each real edge
				- np.array : Indices for the end of each real edge
				- np.array : Features for each reciprocal edge
				- np.array : Global state for each reciprocal state
				- np.array : Indices for the start of each reciprocal edge
				- np.array : Indices for the end of each reciprocal edge
				- np.array : Index of graph associated with each node
				- np.array : Index of graph associated with each real connection
				- np.array : Index of graph associated with each reciprocal connection
		"""
		# get atom's structure id
		gnode = []
		for i, j in enumerate(x0_temp):
			gnode += [i] * len(j)
		# get bond features from a batch of structures
		# get bond's structure id
		gbond_real = []
		for i, j in enumerate(x1_temp):
			gbond_real += [i] * len(j)

		gbond_reciprocal = []
		for i, j in enumerate(x5_temp):
			gbond_reciprocal += [i] * len(j)

		# assemble atom features together
		n_atoms = [len(i) for i in x0_temp]
		x0_temp = np.concatenate(x0_temp, axis=0)
		x0_temp = self.process_atom_feature(x0_temp)

		# assemble bond feature together
		x1_temp = np.concatenate(x1_temp, axis=0)
		x1_temp = self.process_bond_feature(x1_temp)

		# assemble state feature together
		x2_temp = np.concatenate(x2_temp, axis=0)
		x2_temp = self.process_state_feature(x2_temp)

		# assemble bond indices
		x3 = []
		x4 = []
		offset_ind = 0
		for ind1, ind2, n_atom in zip(x3_temp, x4_temp, n_atoms):
			x3 += [i + offset_ind for i in ind1]
			x4 += [i + offset_ind for i in ind2]
			# offset_ind += max(ind1) + 1
			offset_ind += n_atom

		# Repeat for reciprocal part
		# assemble bond feature together
		x5_temp = np.concatenate(x5_temp, axis=0)
		x5_temp = self.process_bond_feature(x5_temp)

		# assemble state feature together
		x6_temp = np.concatenate(x6_temp, axis=0)
		x6_temp = self.process_state_feature(x6_temp)

		# assemble bond indices
		x7 = []
		x8 = []
		offset_ind = 0
		for ind1, ind2, n_atom in zip(x7_temp, x8_temp, n_atoms):
			x7 += [i + offset_ind for i in ind1]
			x8 += [i + offset_ind for i in ind2]
			# offset_ind += max(ind1) + 1
			offset_ind += n_atom

		# Compile the inputs in needed order
		inputs = (
			expand_1st(x0_temp),
			expand_1st(x1_temp),
			expand_1st(x2_temp),
			expand_1st(np.array(x3, dtype=np.int32)),
			expand_1st(np.array(x4, dtype=np.int32)),
			expand_1st(x5_temp),
			expand_1st(x6_temp),
			expand_1st(np.array(x7, dtype=np.int32)),
			expand_1st(np.array(x8, dtype=np.int32)),
			expand_1st(np.array(gnode, dtype=np.int32)),
			expand_1st(np.array(gbond_real, dtype=np.int32)),
			expand_1st(np.array(gbond_reciprocal, dtype=np.int32)),

		)
		return inputs


import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras import Model
from megnet.utils.layer import repeat_with_index
from tensorflow_addons.activations import mish
from tensorflow.keras.layers import Dense, Add

class _MLPLayer(Layer):
	"""
	A simple series of densely connected layers in a pre-built block.
	Refer to tensorflow Dense documention for further insight.

	Args:
		units_list : List[int]
			List of widths to be used in the MLP block
		activation : str or func
			Choice of activation function for the layers
	"""

	def __init__(self,units_list,activation,**kwargs):
		super(_MLPLayer, self).__init__(**kwargs)
		self.units_list = units_list
		self.activation = activation
		self.layers = [Dense(n_i,activation) for n_i in self.units_list]
		return

	def call(self,inputs):
		output = inputs
		for layer_i in self.layers: output = layer_i(output)
		return output

class ResiMLPLayer(Layer): 
	"""
	A simple series of densely connected layers in a pre-built block.
	Refer to tensorflow Dense documention for further insight.

	Args:
		units_list : List[int]
			List of widths to be used in the MLP block
		activation : str or func
			Choice of activation function for the layers

	"""

	def __init__(self,units,nlayers,activation,**kwargs):
		super(ResiMLPLayer, self).__init__(**kwargs)
		self.units = units
		self.nlayers = nlayers
		self.activation = activation
		self.projection_layer = Dense(units,"linear")
		self.layers = [Dense(units,activation) for _ in range(nlayers)]
		return

	def call(self,inputs):
		output = self.projection_layer(inputs)
		for layer_i in self.layers: 
			output_temp = layer_i(output)
			output = Add()([output_temp, output])
		return output
		
class ContextGCNLayer(Layer):
	"""
	The core Contextual Message passing block for ContextNET.
	   
	Args:
		units_list : List[int]
			List of widths to be used in the ResiMLP blocks within the layer
		activation : str or func
			Choice of activation function for the layers
		reduce_method : str
			Choice of reduction for graph messages. "mean" or "sum" currently supported.

	"""


	def __init__(self,units_list,activation = mish, reduce_method="mean",**kwargs):
		super(ContextGCNLayer, self).__init__(**kwargs)
		self.units_list = units_list
		self.activation = activation
		Dense = tf.keras.layers.Dense
		if reduce_method == "sum":
			self.unsorted_reduce_method = tf.math.unsorted_segment_sum
			self.sorted_reduce_method   = tf.math.segment_sum
		if reduce_method == "max":
			self.unsorted_reduce_method = tf.math.unsorted_segment_max
			self.sorted_reduce_method   = tf.math.segment_max
		else:
			self.unsorted_reduce_method = tf.math.unsorted_segment_mean
			self.sorted_reduce_method   = tf.math.segment_mean

		# Set up the trainable connected
		self.eMLP = ResiMLPLayer(units_list[0], len(units_list), activation)
		self.mMLP = ResiMLPLayer(units_list[0], len(units_list), activation)
		self.vMLP = ResiMLPLayer(units_list[0], len(units_list), activation)
		self.uMLP = ResiMLPLayer(units_list[0], len(units_list), activation)
		return

	def edge_update(self,inputs):
		"""
		Takes in graph, and updates the edges
		based on the state, and each of the verts
		the edges is running between
		"""
		verts, edges, u, index_i, index_j, gvert, gedge = inputs
		index_i = tf.reshape(index_i, (-1,))
		index_j = tf.reshape(index_j, (-1,))
		vi = tf.gather(verts, index_i, axis=1)
		vj = tf.gather(verts, index_j, axis=1)
		concate_verts = tf.concat([vi, vj], -1)
		u_expand = repeat_with_index(u, gedge, axis=1)
		concated = tf.concat([concate_verts, edges, u_expand], -1)
		return self.eMLP(concated)

	def edge_local_context(self,e_ij_updated,inputs):
		"""
		Calculates local edge information on a per-vertex basis, reducing based
		on predefined reduction methods, and returns them in line with given vertex incoming
		edge indices. 
		"""
		verts, edges, u, index_i, index_j, gvert, gedge = inputs
		index_i = tf.reshape(index_i, (-1,))
		return tf.expand_dims(self.unsorted_reduce_method(tf.squeeze(e_ij_updated,axis=0), index_i, num_segments=tf.shape(verts)[1]), axis=0)

	def edge_global_context(self,e_ij_updated,inputs):
		"""
		Calculates global edge information, reducing based on predefined reduction methods, 
		and returns them in line with given vertex incoming edge indices. 
		"""
		verts, edges, u, index_i, index_j, gvert, gedge = inputs
		gedge = tf.reshape(gedge, (-1,))
		return tf.expand_dims(self.sorted_reduce_method(tf.squeeze(e_ij_updated,axis=0), gedge), axis=0)

	def create_messages(self,e_ij_updated, e_i_local, e_global, inputs):
		"""
		Takes in updated edges, the local contexts, and global contexts, and calculates
		each incoming message on a per-edge basis.
		"""
		verts, edges, u, index_i, index_j, gvert, gedge = inputs
		index_i = tf.reshape(index_i, (-1,))
		gedge  = tf.reshape(gedge, (-1,))
		# Gather local and global edge features
		e_i_local = tf.gather(e_i_local, index_i, axis=1)
		e_global  = repeat_with_index(e_global, gedge, axis=1)
		# concatenate each edge with the local and global contexts
		m_ij = tf.concat([e_ij_updated, e_i_local, e_global], -1)
		return self.mMLP(m_ij)

	def vert_update(self,m_ij,inputs):
		"""
		Performs the vertex updates based on calculated messages, returning the updated vertex values.
		"""
		verts, edges, u, index_i, index_j, gvert, gedge = inputs
		index_i = tf.reshape(index_i, (-1,))
		gvert   = tf.reshape(gvert, (-1,))
		m_i     = tf.expand_dims(self.unsorted_reduce_method(tf.squeeze(m_ij, axis=0), index_i, num_segments=tf.shape(verts)[1]), axis=0)
		u_expand = repeat_with_index(u, gvert, axis=1)
		concated = tf.concat([verts, m_i, u_expand], -1)
		return self.vMLP(concated)

	def state_update(self, v_updated, e_ij_updated, inputs):
		"""
		Updates the global state vectors based on the graph edges and vertices, per-graph.
		"""
		verts, edges, u, index_i, index_j, gvert, gedge = inputs
		gedge = tf.reshape(gedge, (-1,))
		u_e = tf.expand_dims(self.sorted_reduce_method(tf.squeeze(e_ij_updated, axis=0), gedge), axis=0)
		gvert = tf.reshape(gvert, (-1,))
		u_v = tf.expand_dims(self.sorted_reduce_method(tf.squeeze(v_updated, axis=0), gvert), axis=0)
		concated = tf.concat([u_e, u_v, u], -1)
		return self.uMLP(concated)

	def call(self,inputs):
		"""
		Perform the contextual message passing.
		"""
		# Edges
		e_ij_updated	= self.edge_update(inputs)
		e_i_local		= self.edge_local_context(e_ij_updated, inputs)
		e_global		= self.edge_global_context(e_ij_updated, inputs)
		# E->V messages
		m_ij = self.create_messages(e_ij_updated, e_i_local, e_global, inputs)
		# Vert update
		v_i = self.vert_update(m_ij, inputs)
		# State update
		u = self.state_update(v_i, e_ij_updated, inputs)
		return [v_i, e_ij_updated, u]

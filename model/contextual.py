import tensorflow as tf
from tensorflow.keras import Input
from megnet.layers import GaussianExpansion
from ContextualMPNN.reci_layers.readout import ReadoutMeanLayer, ReadoutSumLayer, ReadoutMaxLayer
from ContextualMPNN.reci_layers.contextual import  ContextGCNLayer, ResiMLPLayer, ContextGCN_MLP_Layer
from tensorflow_addons.activations import mish

def make_context_model(centers, width, C=64, Nelem = 95, readout="mean", reduce="mean"):
	"""
	 Builds the standard contextual message passing MPNN used for matbench/mp18
	 using only real-space features.
	 Args:
		centers : np.array 
			The centers of the gaussian edge-distance feature expansion. 
		width : float
			The width of the gaussian response to edge-distance features expansion.
		C : int
			Width of the trainable layers used within the model.
		Nelem : int
			The number of chemical elements supported in the embedding layer, (Max Z).
		readout : str
			The type of reduction used in the readout of the model. Supports "mean" or "sum".
		reduce : str
			The type of reduction used in the message passing layers of the model. Supports "mean" or "sum".
	Returns:
		model : tf.keras.Model
			Full contextual MPNN for just realspace.
	"""

	if readout=="mean":
		readout_func = ReadoutMeanLayer()
	elif readout == "sum":
		readout_func = ReadoutSumLayer()
	elif readout == "max":
		readout_func = ReadoutMaxLayer()
	else:
		print("WARNING: Model falling back to mean readout.")
		readout_func = ReadoutMeanLayer()


	# Input setup
	x0 = Input(shape=(None,),   dtype=tf.int32,		name="atom_int_input")
	x1 = Input(shape=(None,),   dtype=tf.float32,	name="bond_float_input_real")
	x2 = Input(shape=(None, 2), dtype=tf.float32,	name="state_default_input_real") # Ignored, but consistent with MEGNET style
	x3 = Input(shape=(None,),   dtype=tf.int32,		name="bond_index_1")
	x4 = Input(shape=(None,),   dtype=tf.int32,		name="bond_index_2")
	x5 = Input(shape=(None,),   dtype=tf.int32,		name="atom_graph_index_input")
	x6 = Input(shape=(None,),   dtype=tf.int32,		name="bond_graph_index_input")

	# Preprocessing of inputs
	x0_ = tf.keras.layers.Embedding(Nelem, C, name="atom_embedding", trainable=True)(x0) #TODO FIX hard-coded element Zmax
	x1_ = GaussianExpansion(centers=centers, width=width)(x1)

	# Linear project to same dimension to make residual connections easier
	x0_ = tf.keras.layers.Dense(C,"linear")(x0_)
	x1_ = tf.keras.layers.Dense(C,"linear")(x1_)
	x2_ = tf.keras.layers.Dense(C,"linear")(x2)

	# Conv layers
	x0u, x1u, x2u = ContextGCNLayer([C,C],reduce_method=reduce)([x0_,x1_,x2_,x3,x4,x5,x6])
	x0u = tf.keras.layers.Add()([x0_, x0u])
	x1u = tf.keras.layers.Add()([x1_, x1u])
	x2u = tf.keras.layers.Add()([x2_, x2u])

	x0u, x1u, x2u = ContextGCNLayer([C,C],reduce_method=reduce)([x0u,x1u,x2u,x3,x4,x5,x6])
	x0u = tf.keras.layers.Add()([x0_, x0u])
	x1u = tf.keras.layers.Add()([x1_, x1u])
	x2u = tf.keras.layers.Add()([x2_, x2u])

	x0u, x1u, x2u = ContextGCNLayer([C,C],reduce_method=reduce)([x0u,x1u,x2u,x3,x4,x5,x6])
	x0u = tf.keras.layers.Add()([x0_, x0u])
	x1u = tf.keras.layers.Add()([x1_, x1u])
	x2u = tf.keras.layers.Add()([x2_, x2u])

	# Pool
	x0u = readout_func([x0u, x5])
	x1u = readout_func([x1u, x6])

	## Readout, old version
	readout = tf.keras.layers.Concatenate(axis=-1)([x0u,x1u,x2u])
	readout = tf.keras.layers.Dense(C, "linear")(readout) # reshape projection
	out = tf.keras.layers.Dense(C, mish)(readout)
	out = tf.keras.layers.Add()([out, readout])
	out = tf.keras.layers.Dense(C, mish)(out)
	out = tf.keras.layers.Add()([out, readout])
	out = tf.keras.layers.Dense(C, mish)(out)
	out = tf.keras.layers.Add()([out, readout])
	
	#out = ResiMLPLayer(2*C, 3, activation=mish)(readout)

	# Final out
	out = tf.keras.layers.Dense(1, "linear")(out)
	model = tf.keras.models.Model(inputs=[x0,x1,x2,x3,x4,x5,x6], outputs=out)
	return model

def make_context_dual_model(centers_real, width_real, centers_reciprocal, width_reciprocal, C=64, Nelem=95):
	"""
	 Builds the standard contextual message passing MPNN used for matbench/mp18
	 using only real-space features.
	 Args:
		centers_real : np.array 
			The centers of the gaussian realspace edge-distance feature expansion. 
		width_real : float
			The width of the gaussian response to realspace edge-distance features expansion.
		centers_reciprocal : np.array 
			The centers of the gaussian reciprocal space edge-distance feature expansion. 
		width_reciprocal : float
			The width of the gaussian response to reciprocal space edge-distance features expansion.
		C : int
			Width of the trainable layers used within the model.
		Nelem : int
			The number of chemical elements supported in the embedding layer, (Max Z)

	Returns:
		model : tf.keras.Model
			Full contextual MPNN for just realspace.
	"""
	# Input setup
	x0 = Input(shape=(None,),   dtype=tf.int32,		name="atom_int_input")
	x1 = Input(shape=(None,),   dtype=tf.float32,	name="bond_float_input_real")
	x2 = Input(shape=(None, 2), dtype=tf.float32,	name="state_default_input_real") # Ignored, but consistent with MEGNET style
	x3 = Input(shape=(None,),   dtype=tf.int32,		name="bond_index_1_real")
	x4 = Input(shape=(None,),   dtype=tf.int32,		name="bond_index_2_real")
	x5 = Input(shape=(None,),   dtype=tf.float32,	name="bond_float_input_reciprocal")
	x6 = Input(shape=(None, 2), dtype=tf.float32,	name="state_default_input_reciprocal") # Ignored, but consistent with MEGNET style
	x7 = Input(shape=(None,),   dtype=tf.int32,		name="bond_index_1_reciprocal")
	x8 = Input(shape=(None,),   dtype=tf.int32,		name="bond_index_2_reciprocal")
	x9 = Input(shape=(None,),   dtype=tf.int32,		name="atom_graph_index_input")
	x10 = Input(shape=(None,),   dtype=tf.int32,	name="bond_graph_index_input_real")
	x11 = Input(shape=(None,),   dtype=tf.int32,	name="bond_graph_index_input_reciprocal")

	# Preprocessing of inputs
	x0r_ = tf.keras.layers.Embedding(Nelem, C, name="atom_embedding_real", trainable=True)(x0) 
	x0k_ = tf.keras.layers.Embedding(Nelem, C, name="atom_embedding_reciprocal", trainable=True)(x0)

	x1_ = GaussianExpansion(centers=centers_real, width=width_real)(x1)
	x5_ = GaussianExpansion(centers=centers_reciprocal, width=width_reciprocal)(x5)

	# Linear project to same dimension to make residual connections easier
	# Real
	x0r_ = tf.keras.layers.Dense(C,"linear")(x0r_)
	x1_ = tf.keras.layers.Dense(C,"linear")(x1_)
	x2_ = tf.keras.layers.Dense(C,"linear")(x2)

	# Conv layers real
	x0ru, x1u, x2u = ContextGCNLayer([C,C])([x0r_,x1_,x2_,x3,x4,x9,x10])
	x0ru = tf.keras.layers.Add()([x0r_, x0ru])
	x1u = tf.keras.layers.Add()([x1_, x1u])
	x2u = tf.keras.layers.Add()([x2_, x2u])

	x0ru, x1u, x2u = ContextGCNLayer([C,C])([x0ru,x1u,x2u,x3,x4,x9,x10])
	x0ru = tf.keras.layers.Add()([x0r_, x0ru])
	x1u = tf.keras.layers.Add()([x1_, x1u])
	x2u = tf.keras.layers.Add()([x2_, x2u])

	x0ru, x1u, x2u = ContextGCNLayer([C,C])([x0ru,x1u,x2u,x3,x4,x9,x10])
	x0ru = tf.keras.layers.Add()([x0r_, x0ru])
	x1u = tf.keras.layers.Add()([x1_, x1u])
	x2u = tf.keras.layers.Add()([x2_, x2u])

	# Conv layers reciprocal
	x0k_ = tf.keras.layers.Dense(C,"linear")(x0k_)
	x5_ = tf.keras.layers.Dense(C,"linear")(x5_)
	x6_ = tf.keras.layers.Dense(C,"linear")(x6)

	x0ku, x5u, x6u = ContextGCNLayer([C,C])([x0k_,x5_,x6_,x7,x8,x9,x11])
	x0ku = tf.keras.layers.Add()([x0k_, x0ku])
	x5u = tf.keras.layers.Add()([x5_, x5u])
	x6u = tf.keras.layers.Add()([x6_, x6u])

	x0ku, x5u, x6u = ContextGCNLayer([C,C])([x0ku,x5u,x6u,x7,x8,x9,x11])
	x0ku = tf.keras.layers.Add()([x0k_, x0ku])
	x5u = tf.keras.layers.Add()([x5_, x5u])
	x6u = tf.keras.layers.Add()([x6_, x6u])

	x0ku, x5u, x6u = ContextGCNLayer([C,C])([x0ku,x5u,x6u,x7,x8,x9,x11])
	x0ku = tf.keras.layers.Add()([x0k_, x0ku])
	x5u = tf.keras.layers.Add()([x5_, x5u])
	x6u = tf.keras.layers.Add()([x6_, x6u])

	# Pool
	# Atoms
	x0ru = ReadoutMeanLayer()([x0ru, x9])
	x0ku = ReadoutMeanLayer()([x0ku, x9])
	# Bonds
	x1u = ReadoutMeanLayer()([x1u, x10])
	x5u = ReadoutMeanLayer()([x5u, x11])

	readout = tf.keras.layers.Concatenate(axis=-1)([x0ru,x1u,x2u,x0ku,x5u,x6u])
	readout = tf.keras.layers.Dense(2*C, "linear")(readout) # reshape projection

	# Readout
	out = tf.keras.layers.Dense(2*C, mish)(readout)
	out = tf.keras.layers.Add()([out, readout])
	out = tf.keras.layers.Dense(2*C, mish)(out)
	out = tf.keras.layers.Add()([out, readout])
	out = tf.keras.layers.Dense(2*C, mish)(out)
	out = tf.keras.layers.Add()([out, readout])
	
	# Final out
	out = tf.keras.layers.Dense(1, "linear")(out)
	model = tf.keras.models.Model(inputs=[x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11], outputs=out)
	return model


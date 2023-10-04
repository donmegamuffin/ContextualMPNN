import tf.tensorflow as tf

class ReadoutSumLayer(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(ReadoutSumLayer, self).__init__(**kwargs)
		return

	def call(self,input_data):
		"""
		Feature sum readout function

		Args:
			feats : tf.tensor
				Feature tensor
			gindex : tf.tensor
				Graph-level indices per feature

		Returns: 
			tf.tensor
				Features sum-reduced per graph
		"""
		feats,gindex = input_data
		temp = tf.expand_dims(tf.math.segment_sum(tf.squeeze(feats,axis=0),tf.squeeze(gindex)),axis=0)
		return temp

class ReadoutMaxLayer(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(ReadoutMaxLayer, self).__init__(**kwargs)
		return

	def call(self,input_data):
		"""
		Feature max readout function

		Args:
			feats : tf.tensor
				feature tf.tensor
			gindex : tf.tensor
				graph-level indices per feature

		Returns: 
			tf.tensor
				Features max-reduced per graph
		"""
		feats,gindex = input_data
		temp = tf.expand_dims(tf.math.segment_max(tf.squeeze(feats,axis=0),tf.squeeze(gindex)),axis=0)
		return temp

class ReadoutMeanLayer(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(ReadoutMeanLayer, self).__init__(**kwargs)
		return

	def call(self,input_data):
		"""
		Feature max readout function

		Args:
			feats : tf.tensor
				feature tf.tensor
			gindex : tf.tensor
				graph-level indices per feature

		Returns: 
			tf.tensor
				Features mean-reduced per graph
		"""
		feats,gindex = input_data
		temp = tf.expand_dims(tf.math.segment_mean(tf.squeeze(feats,axis=0),tf.squeeze(gindex)),axis=0)
		return temp

class ReadoutCombinedLayer(tf.keras.layers.Layer):
	def __init__(self,**kwargs):
		super(ReadoutCombinedLayer,self).__init__(**kwargs)
		return

	def readout_sum(self,input_data):
		"""
		Feature sum readout function

		Args:
			feats : tf.tensor
				Feature tensor
			gindex : tf.tensor
				Graph-level indices per feature

		Returns: 
			tf.tensor
				Features sum-reduced per graph
		"""
		feats,gindex = input_data
		temp = tf.expand_dims(tf.math.segment_sum(tf.squeeze(feats),tf.squeeze(gindex)),axis=0)
		temp.set_shape((None,None,feats.shape[-1]))
		return temp

	def readout_max(self,input_data):
		"""
		Feature max readout function

		Args:
			feats : tf.tensor
				feature tf.tensor
			gindex : tf.tensor
				graph-level indices per feature

		Returns: 
			tf.tensor
				Features max-reduced per graph
		"""
		feats, gindex = input_data
		temp = tf.expand_dims(tf.math.segment_max(tf.squeeze(feats),tf.squeeze(gindex)),axis=0)
		temp.set_shape((None,None,feats.shape[-1]))
		return temp

	def readout_mean(self,input_data):
		"""
		Feature max readout function

		Args:
			feats : tf.tensor
				feature tf.tensor
			gindex : tf.tensor
				graph-level indices per feature

		Returns: 
			tf.tensor
				Features mean-reduced per graph
		"""
		feats, gindex = input_data
		temp = tf.expand_dims(tf.math.segment_mean(tf.squeeze(feats),tf.squeeze(gindex)),axis=0)
		temp.set_shape((None,None,feats.shape[-1]))
		return temp

	def readout_min(self,input_data):
		"""
		Feature min readout function

		Args:
			feats : tf.tensor
				feature tf.tensor
			gindex : tf.tensor
				graph-level indices per feature

		Returns: 
			tf.tensor
				Features min-reduced per graph
		"""
		feats, gindex = input_data
		temp = tf.expand_dims(tf.math.segment_min(tf.squeeze(feats),tf.squeeze(gindex)),axis=0)
		temp.set_shape((None,None,feats.shape[-1]))
		return temp

	def call(self,input_data):
		"""
		Applies feature reduction into single feature comprised of
		four different permutation invariant functions concatenated
		together: sum, max, mean, and min.

		Args:
			feats : tf.tensor
				feature tf.tensor
			gindex : tf.tensor
				graph-level indices per feature

		Returns: 
			tf.tensor
				Readout feature vector per-graph.
		"""
		ro_sum = self.readout_sum(input_data)
		ro_max = self.readout_max(input_data)
		ro_mean= self.readout_mean(input_data)
		ro_min = self.readout_min(input_data)
		return tf.concat([ro_sum,ro_max,ro_mean,ro_min],-1)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:11:37 2023

@author: vsbh19
The following is a clustering layer that will be added to the deep embedded
clustering network 
"""
import tensorflow as tf
from keras.layers import Layer, InputSpec
from keras import backend as K
class ClusteringLayer(Layer):
    """Below is the clustering layer for each of the models calculated using the t distribution (similar to normal 
    but with fatter tails)
    
    Input = 2D tensor with shape (number_samples, n_features)
    Output = 2D tensor with shape (number_samples, n_clusters)
    
    Arguments 
        n_clusters = number clusters
        weights : Numpy array with shape (n_clusters, n_features) -> initial cluster centres
        alpha = t-student distribution parameter which defaults to 0 
    
    
    """
    
    #Below is the constructor, it is where the variables are set for the class
    def __init__ (self, n_clusters, weights = None, alpha = 1.0, **kwargs):
        
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim")) #converts input dim to shape
        super(ClusteringLayer, self).__init__(**kwargs) #call the constructor of the parents class
        #This below sets parameters within the "self"
        self.n_clusters  = n_clusters
        self.alpha = alpha 
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim =2)
    #Time to construct the layer and its weigths
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype = K.floatx(), shape = (None, input_dim))#
        print(self.n_clusters, input_dim)
        self.clusters = self.add_weight(shape = (self.n_clusters, input_dim), initializer='glorot_uniform', name='clustering')
        
        if self.initial_weights is not None: 
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
    
    #calulcats student's t distribution probability density function? 
    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) 
        return q
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

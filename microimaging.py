# Machine Learning and Visualization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import glorot_uniform
from vis.utils import utils
from vis.visualization import visualize_cam

# Image Generation and Plotting
import matplotlib.pyplot as plt

# Utility Library
import numpy as np

def plot_map(gradient):
	fig, axes = plt.subplots(1,2, figsize=(14,5))
	axes[0].imshow(_img)
	axes[1].imshow(_img)
	i = axes[1].imshow(grads,cmap="jet",alpha=0.8)
	fig.colorbar(i)
	plt.suptitle("MRI of Dementia heatmap")


def generate_grad(model, img, flat_sorted_class, flat_sorted_layer):
	grad_layer = utils.find_layer_idx(model, "dense_1")
	grad_class = flat_sorted_class[0]
	copy_img = img
	final_grad = visualize_cam(model, flat_sorted_layer, flat_sorted_class, copy_img, penultimate_layer_idx = grad_layer, backprop_modifier = None, grad_modifier = None)
	plot_map(final_grad)


def grad_viewer(model, img, prediction):
	flat_sorted_class = np.argsort(prediction.flatten())[::-1]
	flat_sorted_layer = utils.find_layer_idx(model, 'dense_2')
	model.layers[flat_sorted_layer].activation = keras.activations.linear
	model = utils.apply_modifications(model)
	generate_grad(model, img, flat_sorted_class, flat_sorted_layer)
	

	
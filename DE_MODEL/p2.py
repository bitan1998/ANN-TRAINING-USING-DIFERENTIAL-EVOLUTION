import numpy as np


#ann design
def ann(combied_array_of_relations, nodes_in_hidden_layer, feature_1, feature_2, feature_3):
	ann_outputs = np.zeros(171)
	for j in range(0, 171):	
		node = np.zeros(nodes_in_hidden_layer + 4)
		node[0] = feature_1[j]
		node[1] = feature_2[j]
		node[2] = feature_3[j]
		weight_number = 0
		for i in range(3, (nodes_in_hidden_layer + 3)):
			node[i] = (node[0] * combied_array_of_relations[weight_number]) + (node[1] * combied_array_of_relations[weight_number + nodes_in_hidden_layer]) + (node[2] * combied_array_of_relations[weight_number + 2 * nodes_in_hidden_layer])
			weight_number+=1
		for i in range(3, (nodes_in_hidden_layer + 3)):
			node[nodes_in_hidden_layer + 3] = node[nodes_in_hidden_layer + 3] + (node[i] * combied_array_of_relations[weight_number])
			weight_number+=1
		ann_outputs[j] = node[nodes_in_hidden_layer + 3]
		ann_outputs[j] = sigmoid(ann_outputs[j])
	return ann_outputs
	

#activation function
def sigmoid(x):
        return 1/(1+np.exp(-x))

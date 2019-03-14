import numpy as np
import csv
import p2

#size of matrices of each relation of layers
number_of_ann = int(input("Enter the number of ANNs : "))
nodes_in_hidden_layer = int(input("Enter the number of nodes in hidden layer : "))
size_of_relation_1 = (3, nodes_in_hidden_layer)
size_of_relation_2 = (nodes_in_hidden_layer, 1)


#initializing the matrices and arrays with zeros
feature_1 = feature_2 = feature_3 = expected_output = np.zeros((171))
matrix_of_relation_1 = np.zeros((number_of_ann, 3, nodes_in_hidden_layer))
matrix_of_relation_2 = np.zeros((number_of_ann, nodes_in_hidden_layer, 1))
array_of_relation_1 = np.zeros((number_of_ann, (3 * nodes_in_hidden_layer)))
array_of_relation_2 = np.zeros((number_of_ann, nodes_in_hidden_layer))
combined_array_of_relations = np.zeros((number_of_ann, ((3 * nodes_in_hidden_layer) + nodes_in_hidden_layer)))


#creating the actual randomized matrices and arrays with values ranging from 0 to 1
print("initial weights")
for i in range(number_of_ann):
	matrix_of_relation_1[i] = np.random.uniform(0, 1, size_of_relation_1)
	matrix_of_relation_2[i] = np.random.uniform(0, 1, size_of_relation_2)
	array_of_relation_1[i] = matrix_of_relation_1[i].flatten(order = 'C')
	array_of_relation_2[i] = matrix_of_relation_2[i].flatten(order = 'C')
	combined_array_of_relations[i] = np.append(array_of_relation_1[i], array_of_relation_2[i])
	print (combined_array_of_relations[i])


#reading data
i = 0
with open('data.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		feature_1[i] = float(row['field_soil_temp_c'])
		feature_2[i] = float(row['field_air_temp_c'])
		feature_3[i] = float(row['field_rh'])
		expected_output[i] = float(row['field_soil_wc']) 
		i = i + 1
		

#collecting ann outputs for number_of_ann anns with different weights
ann_outputs = np.zeros((number_of_ann,171))
for i in range(number_of_ann):
	ann_outputs[i] = p2.ann(combined_array_of_relations[i], nodes_in_hidden_layer, feature_1, feature_2, feature_3)
	print(" ")
	print("ann outputs for ann " + str(i))
	print(ann_outputs[i])

#calculating errors with actual parameters
ann_errors=np.zeros((number_of_ann,171))
for i in range(number_of_ann):
        ann_errors[i]=np.square(np.subtract(expected_output,ann_outputs[i]))
        print(" ")
        print("ann errors for ann"+ str(i))
        print(ann_errors[i])

#calculating RMSE values for each ANN
rmse=np.zeros(number_of_ann)
for i in range(number_of_ann):
        rmse[i]=np.std(ann_errors[i], dtype=np.float64)
        print(" ")
        print("RMSE value of an ANN"+ str(i))
        print(rmse[i])

smallest=np.zeros((number_of_ann,1))
smallest[i]=np.min(rmse)
print('\nSmallest among the RMSE is:',smallest[i])



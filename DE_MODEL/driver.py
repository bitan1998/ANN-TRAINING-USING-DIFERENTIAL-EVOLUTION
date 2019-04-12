import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def obj(array_of_weights, nodes_in_hidden_layer):
	ann_outputs = np.zeros(3000)
	ann_outputs = ann(array_of_weights, nodes_in_hidden_layer)
	expected_output = np.zeros((3000))
	i = 0
	with open('data.csv') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			expected_output[i] = float(row['field_soil_wc']) 
			i = i + 1
	for i in range(0, 3000):
		ann_outputs[i]=(expected_output[i]-ann_outputs[i])*(expected_output[i]-ann_outputs[i])
		# rmse=math.sqrt(np.sum(ann_outputs)/3000)
		rmse=np.sqrt(mean_squared_error(expected_output,ann_outputs))
	return rmse

def ann(array_of_weights, nodes_in_hidden_layer):
	feature_1 = feature_2 = feature_3 = expected_output = np.zeros((3000))
	i = 0
	with open('data.csv') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			feature_1[i] = float(row['field_soil_temp_c'])
			feature_2[i] = float(row['field_air_temp_c'])
			feature_3[i] = float(row['field_rh'])
			expected_output[i] = float(row['field_soil_wc']) 
			i = i + 1
	ann_outputs = np.zeros(3000)
	for j in range(0, 3000):	
		node = np.zeros(nodes_in_hidden_layer + 4)
		node[0] = feature_1[j]
		node[1] = feature_2[j]
		node[2] = feature_3[j]
		weight_number = 0
		for i in range(3, (nodes_in_hidden_layer + 3)):
			node[i] = (node[0] * array_of_weights[weight_number]) + (node[1] * array_of_weights[weight_number + nodes_in_hidden_layer]) + (node[2] * array_of_weights[weight_number + 2 * nodes_in_hidden_layer])
			weight_number+=1
		for i in range(3, (nodes_in_hidden_layer + 3)):
			node[nodes_in_hidden_layer + 3] = node[nodes_in_hidden_layer + 3] + (node[i] * array_of_weights[weight_number])
			weight_number+=1
		ann_outputs[j] = node[nodes_in_hidden_layer + 3]
		ann_outputs[j] = 1/(1+np.exp(-ann_outputs[j]))
	return ann_outputs

def de(bounds, popsize, hidden_nodes, mut=0.8, crossp=0.7, its=50):
        dimensions = len(bounds)
        pop = np.random.rand(popsize, dimensions)
        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        fitness = np.asarray([obj(ind,hidden_nodes) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]
        ret_fit = np.zeros(its)
        for i in range(its):
                for j in range(popsize):
                        idxs = [idx for idx in range(popsize) if idx != j]
                        a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
                        mutant = np.clip(a + mut * (b - c), 0, 1)
                        cross_points = np.random.rand(dimensions) < crossp
                        if not np.any(cross_points):
                                cross_points[np.random.randint(0, dimensions)] = True
                        trial = np.where(cross_points, mutant, pop[j])
                        trial_denorm = min_b + trial * diff
                        f = obj(trial_denorm,hidden_nodes)
                        if f < fitness[j]:
                                fitness[j] = f
                                pop[j] = trial
                                if f < fitness[best_idx]:
                                        best_idx = j
                                        best = trial_denorm
                ret_fit[i]=fitness[best_idx];
        return ret_fit, best


number_of_ann = int(input("Enter the number of ANNs : "))
nodes_in_hidden_layer = int(input("Enter the number of nodes in hidden layer : "))

f,b = de(bounds=[(-5,5)] * (3 * nodes_in_hidden_layer + nodes_in_hidden_layer) , popsize=number_of_ann, hidden_nodes=nodes_in_hidden_layer)
plt.xlabel('ITERATION')
plt.ylabel('FITNESS')
plt.title('Evolution of fitness on 500 iterations')
plt.grid()
plt.plot(f)
plt.show()
output = ann(b, nodes_in_hidden_layer)
rmse_output=obj(b,nodes_in_hidden_layer)
print(rmse_output)
target = np.zeros((3000))
i = 0
with open('data.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		target[i] = float(row['field_soil_wc']) 
		i = i + 1
plt.xlabel('target')
plt.ylabel('output')
p1=np.polyfit(target,output,1)
plt.plot(output,np.polyval(p1,output),'r:')
plt.scatter(output,target,facecolors="none", edgecolors="b")
plt.plot(target,target)
plt.show()

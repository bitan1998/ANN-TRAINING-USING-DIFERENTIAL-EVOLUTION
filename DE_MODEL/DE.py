import numpy as np
import matplotlib.pyplot as plt
import p1

def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=40, its=500):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
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
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]

def fobj(x):
    value=0
    for i in range (len(x)):
        value+=x[i]**2
        return value/len(x)



for d in [p1.smallest]:
    it = list(de(lambda x: sum(x**2), bounds=[(-100,100)] *5))
    x,f =zip(*it)
    plt.xlabel('ITERATION')
    plt.ylabel('FITNESS')
    plt.title('Evolution of fitness on 500 iterations')
    plt.ylim((-100,5000))
    plt.xlim((-10,500))
    plt.grid()
    plt.plot(f, label='RMSE_MIN'.format(d))
plt.legend()
plt.show()







from tsp import * 
import numpy as np
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, nPaths=30, survivalRate=65, mutationRate=20, endParameter="epoch", endParameterMax=2000, mutationOperator="inversion"):
        self.nPaths = nPaths
        self.survivalRate = survivalRate
        self.mutationRate = mutationRate
        self.crossoverRate = 100 - (survivalRate + mutationRate)
        self.mutationOperator = self.inversionOperator if mutationOperator == "inversion" else self.swapOperator
        self.end = lambda: (self.epoch >= endParameterMax) if endParameter == "epoch" else  (lambda: self.unchangedIterations >= endParameterMax)  
        self.epoch = 0
        self.unchangedIterations = 0

    @staticmethod
    def swapOperator(path):
        """swap two random genes in the given path/individual/chromosome.

        parameters:
            path (np.ndarray): one solution to the problem.

        returns:
            np.ndarray: the mutated path, with two genes swapped.
        """
        mutation = np.random.choice(len(path), 2, replace=False)  # choosing two random gene indexes
        path[mutation[0]], path[mutation[1]] = path[mutation[1]], path[mutation[0]]  # swapping
        return path

    @staticmethod
    def inversionOperator(path):
        """inverse a random segment of the path/individual/chromosome.

        parameters:
            path (np.ndarray): one solution to the problem.

        returns:
            np.ndarray: the mutated path, with a random segment inverted.
        """
        mutation = np.random.choice(len(path), 2, replace=False) 
        if mutation[0] > mutation[1]:  # outsides of path being inversed 
            righthalf = np.arange(mutation[0], len(path))  # creates array of numbers for remaining right half of array
            lefthalf = np.arange(0, mutation[1] + 1)  # same for the left half up to lower index chosen
            mutationidxs = np.append(righthalf, lefthalf)  # invert the two halves
        else:  # inside of path being inversed
            mutationidxs = np.arange(mutation[0], mutation[1])
        path[mutationidxs] = path[mutationidxs[::-1]] 
        return path

    @staticmethod
    def crossoverOperator(path0, path1):
        """perform crossover between two paths/individuals/chromosomes.

        parameters:
            path0 (np.ndarray): the first parent path.
            path1 (np.ndarray): the second parent path.

        returns:
            np.ndarray: the mutated path resulting from the crossover operation.
        """
        start = np.random.choice(len(path0))
        end = np.random.choice(range(start, len(path0) + 1))
        crossoverstring = path0[start:end] 
        path1 = path1[np.isin(path1, crossoverstring, invert=True)]
        path1 = np.insert(path1, start, crossoverstring)
        return path1

    def __call__(self):
        """execute a genetic algorithm to optimize the tsp solution.

        parameters:
            npaths (int): the number of paths in the population.
            maxunchangediterations (int): the maximum number of iterations without improvement.
            survivalrate (int): the percentage of paths that survive to the next generation.
            mutationrate (int): the percentage of paths that are mutated in the next generation.
            mutationoperator (function): the function used to perform mutations on paths.

        returns:
            none: this function does not return a value but prints results and plots the best route found.
        """
        tsp = TSP()
        paths = np.array([np.random.permutation(tsp.dim) for _ in range(self.nPaths)])
        distances = np.array([tsp(path) for path in paths])
        self.history = []
        self.minDist = np.min(distances)
        self.history.append(self.minDist)
        print(f"initial smallest distance = {self.minDist}\ninitiating genetic algorithm\n")
        while not self.end():
            # best % survives, unique produces the indexes of distances that produce the sorting of unique items, used to choose the surviving paths
            _, sortingidxs = np.unique(distances, return_index=True)
            paths = paths[sortingidxs][:self.nPaths * self.survivalRate // 100]
            # teenage turtles
            while len(paths) < self.nPaths * (self.survivalRate + self.mutationRate) / 100:
                idx = np.random.choice(len(paths))
                mutatedpath = self.mutationOperator(paths[idx].copy())
                paths = np.append(paths, [mutatedpath], axis=0)
            # perform crossovers in the original surviving paths
            crossovers = [np.random.choice(self.nPaths * self.survivalRate // 100, 2, replace=False) for _ in range(self.nPaths * self.crossoverRate // 100)]
            for crossover in crossovers:
                crossoveredpath = self.crossoverOperator(paths[crossover[0]].copy(), paths[crossover[1]].copy())
                paths = np.append(paths, [crossoveredpath], axis=0)
            # calculate new distances and keep track if improvements are being made
            distances = np.array([tsp(path) for path in paths])
            newMin = np.min(distances)
            if newMin < self.minDist:
                self.mindist = newMin
                self.unchangedIterations = 0
            else:
                self.unchangedIterations += 1
            # print status once every 50 epochs and update epoch
            if (not self.epoch % 50):
                print(f'epoch {self.epoch}...')
                print(f'smallest distance = {newMin}')
            self.epoch += 1
            self.history.append(newMin)
        self.bestRoute = paths[np.argmin(distances)]
        self.bestDistance = np.min(distances)
    def plotPath(self):
        with TSP(plot=True) as tsp:
            tsp.plot_route(self.bestRoute, self.bestDistance)
    def plotConvergence(self, label='', xInterval=1):
        plt.plot(range(0, len(self.history)*xInterval, xInterval), self.history, label=label)


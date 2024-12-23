from ga import GeneticAlgorithm as GA
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def make_convergence_plots(upper_limit, repetitions=6, smoothing_window=15):
#     """Makes the plot with the conversion curves"""
    print("Starting Random Search Experiment...")
    average_history = np.zeros(upper_limit)
    for repetition in range(repetitions):
        print(f"repetion={repetition}")
        random_search = RS(upper_limit, plot=False)
        random_search()
        average_history += (random_search.convergence_history - average_history)/(repetition+1)
    plt.plot(savgol_filter(average_history, smoothing_window, 1), label='Random Search')

    nPaths = 30
    survivalRate=65
    changesPerEpoch = int(np.ceil((nPaths*(1-survivalRate/100))))
    print("\n\nStarting Genetic Algorithm with Swap Operator Experiment...")
    mutationOperator="swap"
    average_history = np.zeros(int(np.ceil(upper_limit/changesPerEpoch))+1)
    for repetition in range(repetitions):
        print(f"repetion={repetition}")
        genetic_algorithm = GA(nPaths = nPaths, survivalRate=survivalRate, endParameterMax=np.ceil(upper_limit/changesPerEpoch), mutationOperator=mutationOperator)
        genetic_algorithm()
        average_history += (genetic_algorithm.history - average_history)/(repetition+1)
    plt.plot(range(0,len(average_history)*changesPerEpoch,changesPerEpoch), savgol_filter(average_history, smoothing_window, 1), label={"Genetic Algorithm w/ swap"})

    print("\n\nStarting Genetic Algorithm with Inversion Operator Experiment...")
    mutationOperator="inversion"
    average_history = np.zeros(int(np.ceil(upper_limit/changesPerEpoch))+1)
    for repetition in range(repetitions):
        print(f"repetion={repetition}")
        genetic_algorithm = GA(nPaths = nPaths, survivalRate=survivalRate, endParameterMax=np.ceil(upper_limit/changesPerEpoch), mutationOperator=mutationOperator)
        genetic_algorithm()
        average_history += (genetic_algorithm.history - average_history)/(repetition+1)
    plt.plot(range(0,len(average_history)*changesPerEpoch,changesPerEpoch), savgol_filter(average_history, smoothing_window, 1), label={"Genetic Algorithm w/ inversion"})
    plt.legend()
    plt.xlabel("number of operations")
    plt.ylabel("distance of the best path")
    plt.title("Conversion Plots")
    plt.show()

if __name__ == "__main__":
    # Run Random Search Experiment
    upper_limit = 15000 # set equal to max_unchanged_iterations for GA for fair assessment 
    make_convergence_plots(upper_limit)

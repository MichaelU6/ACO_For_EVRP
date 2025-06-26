import math
import random
import matplotlib.pyplot as plt
from EVRP import *
# max_trials = 30
class Stats():
    def __init__(self, EVRP):     
        self.log_performance = None
        self.perf_filename = None
        self.perf_of_trials = None
        self.MAX_TRIALS = 20
        self.EVRP = EVRP									

    def open_stats(self):
        self.perf_of_trials = [0.0] * self.MAX_TRIALS
        self.perf_filename = f"stats.{self.EVRP.problem_instance}.txt"
        self.log_performance = open(self.perf_filename, "a")

    def get_mean(self, r, value):
        self.perf_of_trials[r] = value

    def mean(self, values):
        return sum(values) / len(values)

    def stdev(self, values, average):
        if len(values) <= 1:
            return 0.0
        dev = sum((x - average) ** 2 for x in values)
        return math.sqrt(dev / (len(values) - 1))

    def best_of_vector(self, values):
        return min(values)

    def worst_of_vector(self, values):
        return max(values)

    def close_stats(self):

        perf_mean_value = self.mean(self.perf_of_trials)
        perf_stdev_value = self.stdev(self.perf_of_trials, perf_mean_value)

        self.log_performance.write(f"\n")
        self.log_performance.write(f"Mean {perf_mean_value:.2f}\t ")
        self.log_performance.write(f"\tStd Dev {perf_stdev_value:.2f}\t \n")
        self.log_performance.write(f"Min: {self.best_of_vector(self.perf_of_trials)}\t \n")
        self.log_performance.write(f"Max: {self.worst_of_vector(self.perf_of_trials)}\t \n")

        self.log_performance.close()
        self.plot_stats()

    def plot_stats(self):
        """Vytvorenie grafu výkonu."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.MAX_TRIALS), self.perf_of_trials, marker='o', linestyle='-', color='b', label='Performance')
        plt.title('ACO Performance over Trials')
        plt.xlabel('Trial Number')
        plt.ylabel('Performance Value')
        plt.axhline(self.mean(self.perf_of_trials), color='r', linestyle='--', label='Mean Performance')
        plt.legend()
        plt.grid()
        plt.savefig(f"performance.png")
        plt.close()

    def free_stats(self):
        del self.perf_of_trials
        
    def plot_stats_forOne(self, max, nodes):
        """Vytvorenie grafu výkonu."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(max), nodes, marker='o', linestyle='-', color='b', label='Performance')
        plt.title('ACO Performance over Trials')
        plt.xlabel('Trial Number')
        plt.ylabel('Performance Value')
        plt.axhline(self.mean(nodes), color='r', linestyle='--', label='Mean Performance')
        plt.legend()
        plt.grid()
        plt.savefig(f"performance_plot.png")
        plt.close()
    
    
    def heatmapMatrix(self, matrix, filename="heatmap.png"):
        plt.figure(figsize=(10, 8))  # jednotná veľkosť
        plt.imshow(matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label="Matrix")
        plt.title("HeatMap")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_convergence(self, values, filename="convergence.png"):
        print(values)
        plt.figure(figsize=(10, 8))  # jednotná veľkosť
        plt.plot(range(1, len(values) + 1), values, marker='o', linestyle='-', color='b', label='Hodnoty')
        mean_value = sum(values) / len(values)
        plt.axhline(mean_value, color='r', linestyle='--', label=f'Priemer: {mean_value:.2f}')
        plt.title('Konvergencia hodnôt')
        plt.xlabel('Iterácia')
        plt.ylabel('Hodnota')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_tour(self, best_solution, node_list, filename="tour_plot.png"):
        node_coords = {node['id']: (node['x'], node['y']) for node in node_list}
        plt.figure(figsize=(10, 8))  # jednotná veľkosť
        
        for i, sub_tour in enumerate(best_solution['tour']):
            x_coords = [node_coords[node_id][0] for node_id in sub_tour]
            y_coords = [node_coords[node_id][1] for node_id in sub_tour]
            plt.plot(x_coords, y_coords, marker='o', label=f"Sub-tour {i+1}")
            for node_id in sub_tour:
                x, y = node_coords[node_id]
                plt.text(x, y, str(node_id), fontsize=9, ha='right')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Visualization of Tours')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

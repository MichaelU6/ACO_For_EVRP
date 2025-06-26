import sys
from EVRP import *
from stats import *
from heuristic import *
import numpy as np

def start_run(r, EVRP):
    """Inicializuje beh heuristického algoritmu."""
    random_seed = r
    EVRP.init_evals()
    EVRP.init_current_best()
    print(f"Run: {r} with random seed {random_seed}")

def end_run(r, EVRP, Stats, Heu, best):
    """Získava pozorovanie behu heuristického algoritmu."""
    #current_best = EVRP.get_current_best()
    #evaluations = EVRP.get_evals()
    Stats.get_mean(r - 1, best[0]['tour_length'])
    print(f"End of run {r} with best solution quality {best[0]['tour_length']}\n")
    #print(f"bets:{Heu.best_sol}")

def main():
    """Hlavná funkcia."""
    global problem_instance
    # Krok 1
    problem_instance = sys.argv[1]  # predať názov súboru .evrp ako argument
    evrp = EVRP()
    evrp.read_problem(problem_instance)

    # Krok 2
    stats = Stats(evrp)
    stats.open_stats()
    start_run(1, evrp)
    
    najlepsie = float("inf")
    najlepsieHodnoty = None
        
    for run in range(20):
        # Krok 3
        heuristic = Heuristic(evrp)
        best, konvergencie, heat = heuristic.run_aco(stats)

        #stats.plot_convergence(konvergencie)
        #stats.heatmapMatrix(heat)
        #stats.plot_tour(best[0], evrp.node_list)
        if best[0]['tour_length'] < najlepsie:
            najlepsieHodnoty = best[0], konvergencie, heat
            najlepsie = best[0]['tour_length']
        # Krok 5
        end_run(run, evrp, stats, heuristic, best)
    #print(najlepsieHodnoty[0]['tour_length'])
    #stats.plot_convergence(najlepsieHodnoty[1])
    #stats.heatmapMatrix(najlepsieHodnoty[2])
    #stats.plot_tour(najlepsieHodnoty[0], evrp.node_list)
    # Krok 6
    stats.close_stats()

    # Uvoľnenie pamäte
    stats.free_stats()
    #heuristic.free_heuristic()
    evrp.free_EVRP()


if __name__ == "__main__":
    main()

def end_test_run(results, subcolony, deposit, evaporation_rate, alpha, beta, best):
    if best == []:
        results.append(f"Sub {subcolony}, Dep {deposit} Rate {evaporation_rate} Alpha {alpha} Beta {beta} Length neexistuje\n")
        #print(f"Sub {subcolony}, Dep {deposit} Rate {evaporation_rate} Alpha {alpha} Beta {beta} Lenght {best[0]['tour_length']}\n")
    else:
        results.append(f"Sub {subcolony}, Dep {deposit} Rate {evaporation_rate} Alpha {alpha} Beta {beta} Length {best[0]['tour_length']}\n")

    
def test():
    """Hlavná funkcia."""
    global problem_instance
    results = []
    # Krok 1
    problem_instance = sys.argv[1]  
    evrp = EVRP()
    evrp.read_problem(problem_instance)

    # Krok 2
    stats = Stats(evrp)
    stats.open_stats()
    start_run(1, evrp)
       
    # Krok 3
    for subcolony in range(1, 2):
        for deposit in np.arange(1, 6.1, 1):
            for evaporation_rate in np.arange(0.2, 0.6, 0.1):
                for alpha in range(1, 6, 1):
                    for beta in range(2, 6, 1):
                        heuristic = Heuristic(evrp, alpha, beta, evaporation_rate, deposit, subcolony)
                        best = heuristic.run_aco(stats)[0]
                        
                        end_test_run(results, subcolony, deposit, evaporation_rate, alpha, beta, best)
                        print(f"Konci beta s poctom {beta}")
                    print(f"Konci alpha  s poctom {alpha}")
                print(f"Konci evaporation rate s poctom {evaporation_rate}")
            print(f"Konci deposti s poctom {deposit}")
        print(f"Konci subcolonia s poctom {subcolony}")

    stats.close_stats()

    # Uvoľnenie pamäte
    stats.free_stats()
    evrp.free_EVRP()
    
    # Uloženie výsledkov do súboru
    with open("vysledky.txt", "w") as f:
        f.writelines(results)
    
#if __name__ == "__main__":
#    test()
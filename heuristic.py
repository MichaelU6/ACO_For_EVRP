
import sys
import math
import random
from EVRP import *
from stats import *
import concurrent.futures
import os
import numpy as np
import time


class Heuristic:
    def __init__(self, EVRP, alpha=5.0, beta=4.0, evaporation_rate=0.3, pheromone_deposit=5.0, subcolonies=1, updateSubcolonies=200):
        
        self.previous_best = float('inf')
        self.EVRP = EVRP  # EVRP objekt
        self.num_vehicles = self.EVRP.MIN_VEHICLES
        self.ants = self.EVRP.problem_size//2
        self.findBest = False
        self.reset = False
        self.bestLocalSol = None
        self.alpha = alpha  # Váha feromónov
        self.beta = beta  # Váha viditeľnosti
        self.evaporation_rate = evaporation_rate  # Rýchlosť odparovania feromónov
        self.pheromone_deposit = pheromone_deposit  # Množstvo feromónov na dobrej trase
        self.iterations = self.EVRP.problem_size   # Počet iterácií pre zlepšenie riešení
        self.subcolonies= subcolonies
        self.updateSubcolonies = updateSubcolonies
        

        # Inicializácia feromónovej matice
        self.pheromone_matrix = [[[1.0 for _ in range(EVRP.ACTUAL_PROBLEM_SIZE)] for _ in range(EVRP.ACTUAL_PROBLEM_SIZE)] for i in range(self.subcolonies)]
        self.konvergenciaSubkolonii = [[] for i in range(self.subcolonies)]
        self.best_solutions = [None for i in range(self.subcolonies)]  # Najlepšie riešenie
        
        self.visibility_matrix = [[0.0 for _ in range(EVRP.ACTUAL_PROBLEM_SIZE)] for _ in range(EVRP.ACTUAL_PROBLEM_SIZE)]
        
        # Tabulka pre vzdialenost depa a zakaznikov od nabijaciek
        self.lenghtCharging = [[] for _ in range(self.EVRP.ACTUAL_PROBLEM_SIZE - self.EVRP.NUM_OF_STATIONS)]
        
        #rozdelenie podla vozidiel na zaklade centier
        self.rozdelenie_customer_podla_vehicle = None

        # Naplnenie vzdialeností
        for customer in range(len(self.lenghtCharging)):
            for charging in range(self.EVRP.NUM_OF_CUSTOMERS + 1, self.EVRP.ACTUAL_PROBLEM_SIZE):
                distance = self.EVRP.get_distance(customer, charging)
                self.lenghtCharging[customer].append(distance)

        # Vypočítanie viditeľnosti (inverzná vzdialenosť)
        for i in range(EVRP.ACTUAL_PROBLEM_SIZE):
            for j in range(EVRP.ACTUAL_PROBLEM_SIZE):
                if i != j:
                    if EVRP.get_distance(i, j) != 0:
                        self.visibility_matrix[i][j] = 1.0 / EVRP.get_distance(i, j)
                    else:
                        self.visibility_matrix[i][j] = 1.0 / 0.01
        
        #Nová matica viditelnosti
        #lambda_1, lambda_2 = 0.8, 0.5  # Možno neskôr doladiť

        #for i in range(EVRP.ACTUAL_PROBLEM_SIZE):
        ##    for j in range(EVRP.ACTUAL_PROBLEM_SIZE):
        #        if i != j:
        #            distance = EVRP.get_distance(i, j)
        #            energy_consumption = EVRP.get_energy_consumption(i, j)  # Spotreba energie
        #            if EVRP.is_charging_station(j):
        #                charging_capacity = EVRP.MAX_CAPACITY # 0 ak tam nie je nabíjačka
        #            else:
        #                charging_capacity = 0
                    
        #            visibility = 1.0 / (distance + lambda_1 * energy_consumption + lambda_2 * charging_capacity + 1e-6)
        #            self.visibility_matrix[i][j] = visibility

            
    def initialize(self, subcolonies):
        template = [
        {
            'tour': [[] for _ in range(self.num_vehicles)],
            'steps': [0] * self.num_vehicles,
            'tour_length': math.inf
        }
        ]

        self.best_solutions = [[dict(t) for t in template] for _ in range(subcolonies)]
               
        filtered_nodes = [node for node in self.EVRP.node_list if 0 < node['id'] <= self.EVRP.NUM_OF_CUSTOMERS]
        suradnice = np.array([(node['x'], node['y']) for node in filtered_nodes])
        ids = np.array([node['id'] for node in filtered_nodes])
        
        # Centrum je uzol s ID 0
        centrum = next(node for node in self.EVRP.node_list if node['id'] == 0)
        centrum_suradnice = np.array([centrum['x'], centrum['y']])
        
        # Výpočet uhlov a zoradenie
        uhly = np.arctan2(suradnice[:, 1] - centrum_suradnice[1], suradnice[:, 0] - centrum_suradnice[0])
        zoradene_indexy = np.argsort(uhly)
        zoradene_ids = ids[zoradene_indexy]
        
        # Rozdelenie do sektorov
        sektory = [set() for _ in range(self.num_vehicles)]
        for i, node_id in enumerate(zoradene_ids):
            sektory[i * self.num_vehicles // len(filtered_nodes)].add(node_id)
        self.rozdelenie_customer_podla_vehicle = sektory
                    
    def initialize_heuristic(self):
        return {
            'tour': [[] for _ in range(self.num_vehicles)], 
            'steps': [0] * self.num_vehicles, 
            'tour_length': math.inf
        }
        
    def run_aco(self, stats):
        # meranie času
        start = time.time()
        
        self.initialize(self.subcolonies)
        self.EVRP.init_evals()
        self.writeParameter()

        sols = []
        num_threads = min(self.iterations, os.cpu_count())  # Optimalizácia počtu vlákien

        def run_single_iteration(iteration):
            """Funkcia na spustenie jednej iterácie ACO"""
            solutions = []
            
            # Paralelizácia mravcov v rámci jednej iterácie
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.run_heuristic,iteration%self.subcolonies) for _ in range(self.ants)]
                for future in concurrent.futures.as_completed(futures):
                    solution = future.result()
                    if solution:
                        solutions.append(solution)

            # Ak reset flag, reštartujeme
            if self.reset:
                self.initialize(self.subcolonies)
                self.reset = False

           

            # Paralelizované vylepšenie najlepších riešení
            with concurrent.futures.ThreadPoolExecutor() as executor:
                best_solutions = [future.result() for solutions in concurrent.futures.as_completed(futures)]


            unique_solutions = []
            seen_tour_lengths = set()

            for sol in sorted(best_solutions, key=lambda x: x['tour_length']):
                if sol['tour_length'] not in seen_tour_lengths or len(unique_solutions) < 5:
                    unique_solutions.append(sol)
                    seen_tour_lengths.add(sol['tour_length'])
                
                if len(unique_solutions) == 5:
                    break

            best_solutions = unique_solutions
            
            update = False
            for solution in best_solutions:
                updateCon = self.update_best_solutions(solution, iteration%self.subcolonies)
                if updateCon:
                    update = True
            self.konvergenciaSubkolonii[iteration%self.subcolonies].append(self.best_solutions[iteration%self.subcolonies][0]['tour_length'])
            if update:
                self.update_pheromones(iteration%self.subcolonies)
            
            return iteration, self.best_solutions[iteration%self.subcolonies]

    # Paralelizácia celých iterácií
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Rozdelíme iterácie do dávok po `batch_size` iterácií
            updatePheromones = 0
            i = 0
            for batch_start in range(0, self.iterations, self.subcolonies):
                i+=1
                updatePheromones+=1
                # Spustíme dávku iterácií
                futures = {executor.submit(run_single_iteration, i): i for i in range(batch_start, min(batch_start + self.subcolonies, self.iterations))}
                sol = []
                # Čakáme, kým sa všetky iterácie v dávke dokončia
                for future in concurrent.futures.as_completed(futures):
                    iteration, solutions = future.result()
                    for s in solutions:
                        if s is not False:
                            sol.append(s)
                
                if updatePheromones == self.updateSubcolonies:
                    updatePheromones = 0
                    self.connect_pheromones_matrix()

        best_index = self.find_best_subcolony()   
        if best_index == -1:
            return []  

        end = time.time()  # Koniec merania
        print(f"Čas vykonania: {end - start:.6f} sekúnd")
        
        
        return self.best_solutions[best_index], self.konvergenciaSubkolonii[best_index], self.pheromone_matrix[best_index]

    # Toto je len pre jedneho mravca
    def run_heuristic(self, pheromoneIndex):
        #Spustenie jedneho mravca a viacerými vozidlami.
        solution = self.initialize_heuristic()
        
        customersListIds = list(range(1, self.EVRP.NUM_OF_CUSTOMERS + 1))
        # Vsetci zakaznici idu
        for vehicle_id in range(self.num_vehicles):
            newCustomerCopy, sol = self.construct_solution(vehicle_id, customersListIds, solution, pheromoneIndex)
            solution = sol
            if newCustomerCopy is False or sol is False:
                break
            customersListIds = newCustomerCopy
            
        if len(customersListIds) != 0:
            while len(customersListIds) != 0:
                solution['tour'].append([])
                solution['steps'].append([0])
                vehicle_id += 1
                newCustomerCopy, sol = self.construct_solution(vehicle_id, customersListIds, solution, pheromoneIndex)
                solution = sol
                if newCustomerCopy is False or sol is False:
                    break
                customersListIds = newCustomerCopy
        
        # Ked vygenerujem pre vsetky auta jedno riesenie porovnam ho ci nieje najlepsie. Vlastne je to riesenie jedneho mravca
        if len(customersListIds) == 0:
            flattened_list = [item for sublist in solution['tour'] for item in sublist[:-1]]
            flattened_list.append(0)
            tour_length = self.EVRP.fitness_evaluation(flattened_list)
            solution['tour_length'] = tour_length
            self.findBest = True
        else:
            return False
        return solution

    def construct_solution(self, vehicle_id, customersListIdsDontService, solution, pheromoneIndex):
        # Konštrukcia riešenia pre konkrétne vozidlo.
        
        energy_temp = 0.0
        capacity_temp = 0.0

        solution['steps'][vehicle_id] = 0
        solution['tour'][vehicle_id].append(self.EVRP.DEPOT)  # Začiatok v depe = 0

        nodeForCheckLoop = -1

        i = 1
        # Ked skoncim som v depe zarucene
        while i == 1:
            # Vyberiem podla toho, ktory krok som urobil kde som. Id vrcholu kde som.
            from_node = solution['tour'][vehicle_id][solution['steps'][vehicle_id]]
            
            # Pozriem ci som nemal ten isty vrchol naposledy, ak ano tak som sa dostal do nekonecneho cyklu.
            if from_node == nodeForCheckLoop:
                return customersListIdsDontService, False
            nodeForCheckLoop = from_node
            
            # Ak som uz obsluzil vsetkych zakaznikov, chcem ist do depa a skoncit
            if len(customersListIdsDontService) == 0:
                sol = self.find_path_in_depo(from_node, energy_temp, vehicle_id, solution, pheromoneIndex)
                solution = sol
                i = 0
                break
            
            # Vyberiem ku ktoremu ZAKAZNIKOVI chcem ist
            to_node = self.select_next_node(from_node, customersListIdsDontService, pheromoneIndex, vehicle_id)

            # Získam požiadavky zákazníka a spotrebu energie do najbližšej nabíjačky (tiež môžeme uložiť do premennej)
            customer_demand = self.EVRP.get_customer_demand(to_node)
            min_Battery = self.EVRP.get_energy_consumption(to_node, self.find_nearest_charging_station(to_node))

            # Ak má vozidlo dost kapacity a energie pre plánovaného zákazníka
            if (capacity_temp + customer_demand <= self.EVRP.MAX_CAPACITY and
                energy_temp + self.EVRP.get_energy_consumption(from_node, to_node) + min_Battery <= self.EVRP.BATTERY_CAPACITY):
                capacity_temp += customer_demand
                energy_temp += self.EVRP.get_energy_consumption(from_node, to_node)
                solution['tour'][vehicle_id].append(to_node)
                solution['steps'][vehicle_id] += 1
                customersListIdsDontService.remove(to_node)
            else:
                # Ak nie je dost kapacity alebo energie
                if capacity_temp + customer_demand > self.EVRP.MAX_CAPACITY:
                    # Ak mám dosť energie na návrat do depa
                    if energy_temp + self.EVRP.get_energy_consumption(from_node, self.EVRP.DEPOT) <= self.EVRP.BATTERY_CAPACITY:
                        solution['tour'][vehicle_id].append(self.EVRP.DEPOT)
                        solution['steps'][vehicle_id] += 1
                        i = 0
                    else:
                        # Riešenie pre návrat do depa
                        sol = self.find_path_in_depo(from_node, energy_temp, vehicle_id, solution, pheromoneIndex)
                        solution = sol
                        i = 0
                else:
                    # Ak nie je problém s kapacitou, hľadáme nabíjačku
                    if self.EVRP.is_charging_station(from_node):
                        break
                    charging_station = self.find_charging_station(from_node, self.EVRP.BATTERY_CAPACITY-energy_temp, pheromoneIndex)
                    solution['tour'][vehicle_id].append(charging_station)
                    energy_temp = 0.0
                    solution['steps'][vehicle_id] += 1

        # Skontrolujem riešenie pre vozidlo, či je dokončené
        self.EVRP.check_solution(solution['tour'][vehicle_id][:solution['steps'][vehicle_id] + 1])

        return customersListIdsDontService, solution

    def select_actual_node(self, actualPath, steps):
        if actualPath[steps] > self.EVRP.NUM_OF_CUSTOMERS:
            return self.select_actual_node(actualPath, steps-1)
        else:
            return actualPath[steps]
     
    def select_next_node(self, from_node, candidate_list, pheromoneIndex, vehicle_id):
        rand_value_vehicle = random.uniform(0, 1)
        
        # gemetricke rozdelenie
        #if rand_value_vehicle <= 0.0 and vehicle_id < self.num_vehicles:
        #    sector_nodes = self.rozdelenie_customer_podla_vehicle[vehicle_id]
        #    filtered_candidate_list = [node for node in candidate_list if node in sector_nodes]
            
        #    if not filtered_candidate_list:
        #        filtered_candidate_list = candidate_list  # Ak nič nenájdeme, použijeme pôvodný zoznam
                
        #else:
        #    filtered_candidate_list = candidate_list
        
        # bez geomaetrického rozdelenia
        filtered_candidate_list = candidate_list

        probabilities = []
        total_prob = 0.0
        
        best_node = None
        best_value = -float("inf")
     
        for to_node in filtered_candidate_list:
            pheromone = self.pheromone_matrix[pheromoneIndex][from_node][to_node] ** self.alpha
            visibility = self.visibility_matrix[from_node][to_node] ** self.beta
            prob = pheromone * visibility
            probabilities.append((to_node, prob))
            total_prob += prob
            if prob > best_value:  # Uloženie najlepšieho uzla
                best_value = prob
                best_node = to_node

        if total_prob == 0:  # Ošetrenie prípadu, ak sú všetky pravdepodobnosti nulové
            return candidate_list[0]


        # Semi-greedy výber (deterministický alebo pravdepodobnostný)
        #q = random.uniform(0, 1)
        #q_0 = 0.5  # Pravdepodobnosť deterministického výberu

        #if q <= q_0:
        #    return best_node  # Greedy výber najlepšieho uzla
        
        # Normalizácia pravdepodobností
        probabilities = [(to_node, prob / total_prob) for to_node, prob in probabilities]
        # Softmax normalizacia
        #exp_probs = np.exp([prob for _, prob in probabilities])
        #softmax_probs = exp_probs / np.sum(exp_probs)
        #probabilities = [(to_node, softmax_prob) for (to_node, _), softmax_prob in zip(probabilities, softmax_probs)]
        # Výber uzla na základe náhodnej hodnoty
        random.shuffle(probabilities)

        rand_value = random.uniform(0, 1)
        cumulative_prob = 0.0
        for to_node, prob in probabilities:
            cumulative_prob += prob
            if rand_value <= cumulative_prob:
                return to_node

        return candidate_list[0]  # Default návrat v prípade problému

    def find_nearest_charging_station(self, current_node):      
        charging_station_distances = self.lenghtCharging[current_node]

        # Získame index najmenšej vzdialenosti
        index_of_nearest_station = charging_station_distances.index(min(charging_station_distances))
   
        return len(self.lenghtCharging)+index_of_nearest_station
        
    def find_charging_station(self, current_node, capacity_enenrgy, pheromoneIndex):
        # Získame index najmenšej vzdialenosti
        index_of_valid_station = [charging for charging in range(self.EVRP.NUM_OF_CUSTOMERS + 1, self.EVRP.ACTUAL_PROBLEM_SIZE) if self.EVRP.get_energy_consumption(current_node, charging) <= capacity_enenrgy]

        probabilities = []
        total_prob = 0.0
     
        for to_node in index_of_valid_station:
            pheromone = self.pheromone_matrix[pheromoneIndex][current_node][to_node] ** self.alpha
            visibility = self.visibility_matrix[current_node][to_node] ** self.beta
            prob = pheromone * visibility
            probabilities.append((to_node, prob))
            total_prob += prob

        if total_prob == 0:  # Ošetrenie prípadu, ak sú všetky pravdepodobnosti nulové
            return index_of_valid_station[0]

        # Normalizácia pravdepodobností
        probabilities = [(to_node, prob / total_prob) for to_node, prob in probabilities]

        # Výber uzla na základe náhodnej hodnoty
        rand_value = random.uniform(0, 1)
        cumulative_prob = 0.0
        for to_node, prob in probabilities:
            cumulative_prob += prob
            if rand_value <= cumulative_prob:
                return to_node

        return index_of_valid_station[0]
    
    def find_path_in_depo(self, current_node, energyCapacity, vehicleId, solution, pheromoneIndex):
        # Zistim kolko baterie je treba z vrhcolu kde som do depa
        distance_to_depo = self.EVRP.get_energy_consumption(current_node, self.EVRP.DEPOT)

        # Ak momentalna bateria vozidla staci do depa nastav dalsi bod mojej cesty depo a pridaj krok
        if self.EVRP.BATTERY_CAPACITY - energyCapacity >= distance_to_depo:
            solution['tour'][vehicleId].append(self.EVRP.DEPOT)
            solution['steps'][vehicleId] += 1
        else:
            # Ak nie je dosť energie, nájdi najbližšiu nabíjaciu stanicu. Presun sa nanu a pridaj krok
            nearest_station = self.find_charging_station(current_node, self.EVRP.BATTERY_CAPACITY - energyCapacity, pheromoneIndex)
            solution['tour'][vehicleId].append(nearest_station)
            solution['steps'][vehicleId] += 1
            # Pridam dalsi vrchol depo a pridam krok
            solution['tour'][vehicleId].append(self.EVRP.DEPOT)
            solution['steps'][vehicleId] += 1
        return solution

    def update_pheromones(self, pheromoneIndex):
        # Aktuálna feromónová matica
        pheromone_matrix = np.array(self.pheromone_matrix[pheromoneIndex])
        
        # Odparovanie feromónov
        pheromone_matrix *= (1 - self.evaporation_rate)
        
        # Aktualizácia feromónov podľa najlepších riešení
        bestSolutionWeight = 0
        for best_solution in self.best_solutions[pheromoneIndex]:
            bestSolutionWeight += 0.1
            for vehicle_tour in best_solution['tour']:
                for k in range(len(vehicle_tour) - 1):
                    i, j = vehicle_tour[k], vehicle_tour[k + 1]
                    deposit_value = (self.pheromone_deposit / best_solution['tour_length']) * (1 - bestSolutionWeight)
                    pheromone_matrix[i, j] += deposit_value
                    pheromone_matrix[j, i] += deposit_value

        # Uloženie výsledku späť
        self.pheromone_matrix[pheromoneIndex] = pheromone_matrix.tolist()
                
    def update_best_solutions(self, solution, pheromoneIndex):
        current_best_solutions = self.best_solutions[pheromoneIndex]
        copy_best_solutions = self.best_solutions[pheromoneIndex]
        # Nájdeme globálne najlepšie riešenie zo všetkých subkolónií
        #global_best = min(
        #    (sol for sublist in self.best_solutions for sol in sublist), 
        #    key=lambda x: x['tour_length']
        #)

        # Nájdeme najlepšie riešenie zo súčasnej subkolónie po pridaní nového riešenia
        if solution['tour_length'] < current_best_solutions[-1]['tour_length']:
            current_best_solutions.append(solution)
            self.best_solutions[pheromoneIndex] = sorted(
                current_best_solutions, key=lambda x: x['tour_length']
            )[:5]  # Zachováme iba 5 najlepších

        # Najlepšie riešenie v tejto subkolónii po aktualizácii
        local_best = min(self.best_solutions[pheromoneIndex], key=lambda x: x['tour_length'])

        # Ak je najlepšie riešenie v subkolónii horšie ako globálne najlepšie, vrátime False
        #return local_best['tour_length'] < global_best['tour_length']
        # Ak je nove riesenie po novom najlepsie v danej feromonovej matici
        return local_best['tour_length'] < copy_best_solutions[0]['tour_length'] or local_best['tour_length'] == solution['tour_length']

    def connect_pheromones_matrix(self):
        best_index = self.find_best_subcolony()
        
        # Najlepšia matica ostáva nezmenená
        best_pheromone_matrix = np.array(self.pheromone_matrix[best_index])

        # Najprv resetujeme všetky matice okrem najlepšej na pôvodné hodnoty
        for i in range(self.subcolonies):
            if i != best_index:
                self.pheromone_matrix[i] = np.full_like(best_pheromone_matrix, 1.0).tolist()

        # Potom všetky matice (okrem najlepšej) upravíme podľa najlepšej
        for i in range(self.subcolonies):
            if i != best_index:
                self.pheromone_matrix[i] = (
                    0.6 * np.array(self.pheromone_matrix[i]) + 0.4 * best_pheromone_matrix
                ).tolist()  # Prevod späť na list
    
    def find_best_subcolony(self):
        best_tour_length = float('inf')
        best_index = -1

        # Iterujeme cez všetky subkolónie
        for x, sol in enumerate(self.best_solutions):
            # Nájdeme najlepšiu (najkratšiu) cestu v danej subkolónii
            min_length = min(d['tour_length'] for d in sol)

            # Ak nájdeme lepšiu (kratšiu) cestu, aktualizujeme výsledok
            if min_length < best_tour_length:
                best_tour_length = min_length
                best_index = x

        return best_index
       
    def writeParameter(self): 
        with open("parametre.txt", 'w') as file:
            file.write(f"Zoznam uzlov s id a súradnicami x a y: {self.EVRP.node_list}\n")
            file.write(f"Zoznam s id a požiadavkami zákazníkov: {self.EVRP.cust_demand}\n")
            file.write(f"Nabijacie stanice: {self.EVRP.charging_station}\n")
            file.write(f"Matica vzdialenosti: {self.EVRP.distances}\n")
            file.write(f"Problem size: {self.EVRP.problem_size}\n")
            file.write(f"Energy consuption: {self.EVRP.energy_consumption}\n")
            file.write(f"Depo: {self.EVRP.DEPOT}\n")
            file.write(f"Počet zákazníkov (bez depa): {self.EVRP.NUM_OF_CUSTOMERS}\n")
            file.write(f"Celkový počet zákazníkov, dobíjacích staníc a depa: {self.EVRP.ACTUAL_PROBLEM_SIZE}\n")
            file.write(f"Optimum: {self.EVRP.OPTIMUM}\n")
            file.write(f"NUM_OF_STATIONS: {self.EVRP.NUM_OF_STATIONS}\n")
            file.write(f"Maximálna energia vozidiel: {self.EVRP.BATTERY_CAPACITY}\n")
            file.write(f"Kapacita vozidiel: {self.EVRP.MAX_CAPACITY}\n")
            file.write(f"MIN_VEHICLES: {self.EVRP.MIN_VEHICLES}\n")
            file.write(f"evals: {self.EVRP.evals}\n")
            file.write(f"current_best: {self.EVRP.current_best}\n")

    def local_search(self, solution):
        return self.two_opt(solution, self.EVRP)
    
    def two_opt(self, solution, EVRP):
        def calculate_tour_length(tour):
            # Vypočíta dĺžku trasy
            # NEDA SA TO URYCHLIT???
            length = 0
            for i in range(len(tour) - 1):
                length += EVRP.get_distance(tour[i], tour[i + 1])
            return length

        def is_feasible(new_tour, EVRP):
            # Overí, či je trasa možná (batéria, kapacita a nabíjanie)
            capacity_temp = 0
            energy_temp = 0
            for vehicle_id in range(len(new_tour)):
                for i in range(len(new_tour[vehicle_id]) - 1):
                    from_node = new_tour[vehicle_id][i]
                    to_node = new_tour[vehicle_id][i + 1]
                    capacity_temp += EVRP.get_customer_demand(to_node) if to_node <= EVRP.NUM_OF_CUSTOMERS else 0
                    energy_temp += EVRP.get_energy_consumption(from_node, to_node)

                    # Over kapacitu a batériu
                    if capacity_temp > EVRP.MAX_CAPACITY:
                        return False
                    if energy_temp > EVRP.BATTERY_CAPACITY:
                        # Skontroluj, či je nabíjacia stanica dostupná
                        if not EVRP.is_charging_station(to_node):
                            return False
                        energy_temp = 0  # Reset batérie po nabíjaní
            return True

        best_tour = solution['tour']
        best_length = solution['tour_length']

        improved = True
        while improved:
            improved = False
            for i in range(1, len(best_tour) - 2):
                for j in range(i + 1, len(best_tour) - 1):
                    if j - i == 1:  # Susedné hrany
                        continue
                    # Vymeníme hrany
                    new_tour = best_tour[:i] + best_tour[i:j][::-1] + best_tour[j:]
                    if is_feasible(new_tour, EVRP):  # Over, či je trasa možná
                        new_length = calculate_tour_length(new_tour)
                        if new_length < best_length:
                            best_tour = new_tour
                            best_length = new_length
                            improved = True
                            break
                if improved:
                    break

        solution['tour'] = best_tour
        solution['tour_length'] = best_length
        return solution

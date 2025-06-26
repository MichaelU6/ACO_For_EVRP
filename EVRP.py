import math

class EVRP():
    def __init__(self):
        self.problem_instance = None  # Názov inštancie
        self.node_list = None  # Zoznam uzlov s id a súradnicami x a y. Depo + zakaznici + nabijacky
        self.cust_demand = None  # Zoznam s id a požiadavkami zákazníkov. Depo + zakaznici + nabijacky
        self.charging_station = None # Zoznam s bool hodnotou ci je alebo nieje nabijaci ten bod. Depo(true) + zakaznici(false) + nabijacky(true)
        self.distances = None  # Matica vzdialeností. Zoznam [[], [], ...] . Hladanie napr vzidalenost bodu s indexom 0 od bodu s indexom 3 => self.distances[0][3]
        self.problem_size = None  # Rozmer problému. Depo + zakaznici.
        self.energy_consumption = None # Cislo kolko minie elektriny za jednotku vzdialenosti.

        self.DEPOT = None  # ID depa (0)
        self.NUM_OF_CUSTOMERS = None  # Počet zákazníkov (bez depa)
        self.ACTUAL_PROBLEM_SIZE = None  # Celkový počet zákazníkov, dobíjacích staníc a depa
        self.OPTIMUM = None # Asi zatial najlepsi vysledok z dat
        self.NUM_OF_STATIONS = None # Pocet nabijaciek
        self.BATTERY_CAPACITY = None  # Maximálna energia vozidiel
        self.MAX_CAPACITY = None  # Kapacita vozidiel
        self.MIN_VEHICLES = None # Pocet vozidiel

        self.evals = None  # Počet vyhodnotení
        self.current_best = None  # Aktuálne najlepší výsledok

    def euclidean_distance(self, i, j):
        """Výpočet euklidovskej vzdialenosti medzi dvoma uzlami."""
        xd = self.node_list[i]['x'] - self.node_list[j]['x']
        yd = self.node_list[i]['y'] - self.node_list[j]['y']
        return math.sqrt(xd * xd + yd * yd)


    def compute_distances(self):
        """Výpočet vzdialeností medzi všetkými uzlami."""
        #global distances
        self.distances = [[self.euclidean_distance(i, j) for j in range(self.ACTUAL_PROBLEM_SIZE)] for i in range(self.ACTUAL_PROBLEM_SIZE)]


    def generate_2D_matrix_double(self, n, m):
        """Generovanie 2D matice s typom double."""
        return [[0.0] * m for _ in range(n)]


    def read_problem(self, filename):
        """Načítanie problému zo súboru."""
        #global problem_size, NUM_OF_CUSTOMERS, ACTUAL_PROBLEM_SIZE, node_list, distances, cust_demand, charging_station

        with open(filename, 'r') as fin:
            for line in fin:
                keywords = line.strip().split()
                if not keywords:
                    continue
                if keywords[0] == "DIMENSION:":
                    self.problem_size = int(keywords[1])
                elif keywords[0] == "EDGE_WEIGHT_FORMAT:" and keywords[1] != "EUC_2D":
                    print("not EUC_2D")
                    exit(0)
                elif keywords[0] == "CAPACITY:":
                    self.MAX_CAPACITY = int(keywords[1])
                elif keywords[0] == "VEHICLES:":
                    self.MIN_VEHICLES = int(keywords[1])
                elif keywords[0] == "ENERGY_CAPACITY:":
                    self.BATTERY_CAPACITY = int(keywords[1])
                elif keywords[0] == "ENERGY_CONSUMPTION:":
                    self.energy_consumption = float(keywords[1])
                elif keywords[0] == "STATIONS:":
                    self.NUM_OF_STATIONS = int(keywords[1])
                elif keywords[0] == "OPTIMAL_VALUE:":
                    self.OPTIMUM = float(keywords[1])
                elif keywords[0] == "NODE_COORD_SECTION":
                    if self.problem_size != 0:
                        self.NUM_OF_CUSTOMERS = self.problem_size - 1
                        self.ACTUAL_PROBLEM_SIZE = self.problem_size + self.NUM_OF_STATIONS
                        #self.TERMINATION += self.ACTUAL_PROBLEM_SIZE*25000
                        self.node_list = [None] * self.ACTUAL_PROBLEM_SIZE
                        for i in range(self.ACTUAL_PROBLEM_SIZE):
                            line = next(fin).strip().split()
                            self.node_list[i] = {'id': int(line[0]) - 1, 'x': float(line[1]), 'y': float(line[2])}
                        self.distances = self.generate_2D_matrix_double(self.ACTUAL_PROBLEM_SIZE, self.ACTUAL_PROBLEM_SIZE)
                elif keywords[0] == "DEMAND_SECTION":
                    if self.problem_size != 0:
                        self.cust_demand = [0] * self.ACTUAL_PROBLEM_SIZE
                        self.charging_station = [False] * self.ACTUAL_PROBLEM_SIZE
                        for _ in range(self.problem_size):
                            temp, demand = map(int, next(fin).split())
                            self.cust_demand[temp - 1] = demand
                        for i in range(self.ACTUAL_PROBLEM_SIZE):
                            if i < self.problem_size:
                                self.charging_station[i] = False
                            else:
                                self.charging_station[i] = True
                                self.cust_demand[i] = 0
                elif keywords[0] == "DEPOT_SECTION":
                    self.DEPOT = int(next(fin)) - 1
                    self.charging_station[self.DEPOT] = True
        if self.ACTUAL_PROBLEM_SIZE == 0:
            print("wrong problem instance file")
            exit(1)
        else:
            self.compute_distances()


    def fitness_evaluation(self, routes):
        """Vyhodnotenie kvality riešenia."""
        tour_length = 0
        """
        for j in range(len(routes['steps'])):
            route = routes['tour'][j][:routes['steps'][j]]
            tour_length += sum(self.distances[route[i]][route[i + 1]] for i in range(len(route) - 1))
        """
        for node in range(len(routes)-1):
            tour_length += self.distances[routes[node]][routes[node + 1]]
        #global current_best
        if tour_length < self.current_best:
            self.current_best = tour_length

        #global evals
        self.evals += 1

        return tour_length


    def print_solution(self, routes):
        """Výpis riešenia."""
        print(", ".join(map(str, routes)))


    def check_solution(self, routes):
        """Overenie platnosti riešenia."""
        energy_temp = self.BATTERY_CAPACITY
        capacity_temp = self.MAX_CAPACITY
        distance_temp = 0.0
        for i in range(len(routes) - 1):
            from_node = routes[i]
            to_node = routes[i + 1]
            capacity_temp -= self.get_customer_demand(to_node)
            energy_temp -= self.get_energy_consumption(from_node, to_node)
            distance_temp += self.get_distance(from_node, to_node)
            if capacity_temp < 0.0:
                print("error: capacity below 0 at customer", to_node)
                self.print_solution(routes)
                exit(1)
            if energy_temp < 0.0:
                print("error: energy below 0 from", from_node, "to", to_node)
                self.print_solution(routes)
                exit(1)
            if to_node == self.DEPOT:
                capacity_temp = self.MAX_CAPACITY
            if self.is_charging_station(to_node) or to_node == self.DEPOT:
                energy_temp = self.BATTERY_CAPACITY


    def get_distance(self, from_node, to_node):
        """Získanie vzdialenosti medzi dvoma uzlami."""
        return self.distances[from_node][to_node]


    def get_energy_consumption(self, from_node, to_node):
        """Získanie spotreby energie pri presune medzi dvoma uzlami."""
        return self.energy_consumption * self.distances[from_node][to_node]


    def get_customer_demand(self, customer):
        """Získanie dopytu pre konkrétneho zákazníka."""
        return self.cust_demand[customer]


    def is_charging_station(self, node):
        """Určenie, či je konkrétny uzol dobíjacou stanicou."""
        return self.charging_station[node]


    def get_current_best(self):
        """Získanie aktuálneho najlepšieho výsledku."""
        return self.current_best


    def init_current_best(self):
        """Resetovanie aktuálneho najlepšieho výsledku."""
        self.current_best = float('inf')


    def get_evals(self):
        """Získanie počtu vyhodnotení."""
        return self.evals


    def init_evals(self):
        """Resetovanie počtu vyhodnotení."""
        self.evals = 0


    def free_EVRP(self):
        """Uvoľnenie alokovanej pamäte."""
        del self.node_list
        del self.cust_demand
        del self.charging_station
        del self.distances

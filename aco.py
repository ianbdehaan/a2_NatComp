from tsp import *
import numpy as np

class ACO():
    def __init__(self, nAnts=10, initial_pher=20000, proximity_constant=3, evaporation_constant=0.01, pheromone_constant=100000, alpha=1, beta=1):
        ''' Class that implements a solution to TSP using Ant Colony Optimization
        
        initial_pher: indicates the initial amount of pheromones present on each edge
        
        '''
        self.nAnts = nAnts
        self.initial_pher = initial_pher
        self.C = proximity_constant
        self.evap_constant = evaporation_constant
        self.Q = pheromone_constant
        self.alpha = alpha
        self.beta = beta
        
        self.tsp = TSP(plot=False)
        self.n = self.tsp.dim 
        self.cities = np.array(self.tsp.create_path(range(self.n))[:-1]) # this creates a array of coordinates(lng, lat) of each city starting with leiden at index 0 
        self.pher_prox_map = self.init_PherProxMap()

    @staticmethod
    def haversine_np(lon1, lat1, lon2, lat2):
        """
        vectorized version of the haversine function    

        """
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371
        return c * r

    def init_PherProxMap(self):
        '''Method that constructs a data structure that holds the amount of pheromones and the proximity for each city. 
        Also updates the distances of each edge and updates the number of pheromones the inital pheromones. It also creates and mapping
        of cities to integers. 
        
        Possible data structures:
         -  Dictionary of Dictionaries
         -  Compact 2D Numpy Array (most memory/time efficient)
         -  Symmetrical 3D Numpy Array (n * n * 2)
         
        Implemented: Compact 2D Numpy Array (Using helper function get_idx_PherProxMap)
    
        '''
        n = self.n
        pher_prox_map = np.zeros((n, n, 2)) # store 2 data values (distance and pherormones) per edge
        iv, jv = np.meshgrid(range(n),range(n))
        pher_prox_map[:,:,1] = ACO.haversine_np(self.cities[iv,1],self.cities[iv,0],self.cities[jv,1], self.cities[jv,0]).reshape((n,n))
        pher_prox_map[:,:,0] = self.initial_pher
        return pher_prox_map
        
    def evaporation_update(self):
        """pheremone evaporation update for all edges."""
        self.pher_prox_map[:,:,0] *= (1-self.evap_constant)
    
    def ant_pheremone_update(self):
        """updates the pheremones based on the routes of all the ants. Looks at the route of one ant and updates the pheromones left behind
        along the route based on the length of the route aka quality of route."""   
        for route in self.ant_routes:
            route_length = self.tsp(route)
            pairs = np.array(list(zip(route[:-1],route[1:]))).T
            self.pher_prox_map[pairs[0],pairs[1],0] += self.Q/route_length
            self.pher_prox_map[pairs[1],pairs[0],0] += self.Q/route_length
    
    def calculate_desire(self, ant_location, allowed_cities):
        '''Calculate the desire to go from city i to the allowed citiesl. Returns the desire as a float. 
        Helper method for generete_ant_routes'''
        pher_prox = self.pher_prox_map[allowed_cities,ant_location].T
        return (pher_prox[0]**self.alpha)*((1/pher_prox[1])**self.beta)
    
    def prob_to_go_to_cities(self, ant_location, allowed_cities):
        """Calculates the probability of going to each city that is still allowed in a dictionary, where the length of
        the dict is equal to the allowed_cities.
        ant_location(int): current city ant is located in"""
        # first getting each indivdual desire from current ant location to every available city
        desires = self.calculate_desire(ant_location, allowed_cities)
        # then taking each city and calculating probability. prob = (desire to specific city/sum of desires for all cities)
        probs = desires/sum(desires)
        return probs
    
    def choose_city(self, start_city, allowed_cities, greedy=False):
        """choose a city from the given possible cities and the current start city. 
        Returns the chosen city"""
        probs = self.prob_to_go_to_cities(start_city, allowed_cities)
        if not greedy:
            chosen_city = np.random.choice(allowed_cities,1,p=probs)[0]
        else:
            chosen_city = allowed_cities[np.argmax(probs)]
        return chosen_city
    
    def generate_ant_routes(self):
        ant_routes = []
        for ant in range(self.nAnts):
            ant_routes.append(self.generate_single_route())
        self.ant_routes = ant_routes

    def generate_single_route(self, greedy=False):
        """returns a single route"""
        route = []
        available_cities = list(range(self.n))
        position = np.random.randint(self.n)
        while available_cities: 
            available_cities.remove(position)
            route.append(position)
            if available_cities:
                position = self.choose_city(position, available_cities, greedy)
        # route.append(0)
        return route
            
    def optimize(self, iterations):
        for idx in range(iterations):
            self.generate_ant_routes()
            # update pheromones
            self.ant_pheremone_update()
            self.evaporation_update()
            print(idx)
            self.best_route = self.generate_single_route(greedy = False)
            self.shortest_dist = self.tsp(self.best_route)
            print(f'shortest dist found = {self.shortest_dist}')
        

#testing
if __name__ == "__main__":
    aco_object = ACO()
    aco_object.optimize(1000)



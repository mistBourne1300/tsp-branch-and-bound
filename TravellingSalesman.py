import json
import heapq
import math
import os
import time
from datetime import timedelta
import argparse
import urllib.parse

import googlemaps
import numpy as np
from API_KEY import API_KEY
from tqdm.auto import tqdm

np.seterr(invalid="ignore")


class TSPNode:
    def __init__(self, matrix, bound=0, infidx=None, path=[], depth=0):
        self.depth = depth
        self.matrix = matrix.copy()
        self.infidx = infidx
        self.path = path.copy()
        self.bound = bound

        if infidx:
            self.path.append(infidx)
        self.__recalibrate__()
        self.route = self.__route__()

    def __recalibrate__(self):
        if self.infidx:
            row, col = self.infidx
            self.matrix[row, :] = np.inf
            self.matrix[:, col] = np.inf
            self.matrix[col, row] = np.inf

        min_rows = np.nanmin(self.matrix, axis=1).reshape((-1, 1))
        self.matrix -= min_rows
        min_cols = np.nanmin(self.matrix, axis=0)
        self.matrix -= min_cols
        self.bound += (
            np.ma.masked_invalid(min_cols).sum() + np.ma.masked_invalid(min_rows).sum()
        )
        self.matrix[np.isnan(self.matrix)] = np.inf

    def valid_moves(self):
        n, m = self.matrix.shape
        if self.infidx:
            i = self.infidx[1]
            for j in range(m):
                if self.matrix[i, j] < np.inf:
                    if j in self.route:
                        continue
                    yield i, j
        else:
            for i in range(n):
                for j in range(m):
                    if self.matrix[i, j] < np.inf:
                        if j in self.route:
                            continue
                        yield i, j

    def children(self):
        for i, j in self.valid_moves():
            yield TSPNode(
                self.matrix,
                bound=self.bound + self.matrix[i, j],
                infidx=(i, j),
                path=self.path,
                depth=self.depth + 1,
            )

    def __route__(self):
        if len(self.path) == 0:
            return []
        route = []
        route.append(self.path[0][0])
        for idx in self.path:
            route.append(idx[1])
        return route

    def soln(self):
        for i in range(len(self.matrix)):
            if i not in self.route:
                return [], False
        return self.route, True

    def __str__(self):
        return (
            str(self.matrix)
            + "\n bound: "
            + str(self.bound)
            + "\npath: "
            + str(self.path)
        )

    def __lt__(self, other):
        # if self.bound == other.bound:
        # 	return self.depth > other.depth
        # return self.bound < other.bound
        if self.depth == other.depth:
            return self.bound < other.bound
        return self.depth > other.depth


class TSPSolver:
    def __init__(self, matrix):
        a, b = matrix.shape
        assert a == b
        self.matrix = matrix
        self.n = a
        self.cities = [i for i in range(self.n)]  # don't think this is necessary

    def costTo(self, i, j):
        return self.matrix[i, j]

    def get_cost(self, route):
        cost = 0
        for i in range(len(route) - 1):
            cost += self.costTo(route[i], route[i + 1])
        return cost + self.costTo(route[-1], route[0])

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        found_tour = False
        total = 0
        bssf = None
        bssf_cost = np.inf
        start_time = time.time()
        while not found_tour and time.time() - start_time < time_allowance:
            route = np.random.permutaion(self.n)
            cost = self.get_cost(route)
            total += 1
            if cost < bssf_cost:
                bssf = route
                bssf_cost = cost
                found_tour = True

        end_time = time.time()
        results["cost"] = bssf_cost
        results["time"] = end_time - start_time
        results["count"] = 1
        results["soln"] = bssf
        results["max"] = None
        results["total"] = total
        results["pruned"] = None
        return results

    def greedy_path(self, start):
        # finds the first cycle, even though that cycle may not visit all cities
        route = [start]
        curr_city = start
        while True:
            # greedily get the next city to visit

            next_city = None
            next_cost = np.inf
            for i in range(self.n):
                if i in route:
                    continue
                cost = self.costTo(curr_city, i)
                if cost < next_cost:
                    next_cost = cost
                    next_city = i

            if next_city == None:
                break
            route.append(next_city)
            cutt_city = next_city
        for i in range(self.n):
            if i not in route:
                return None
        return route

    def greedy(self, time_allowance=60.0):
        results = {}
        count = 0
        bssf = None
        bssf_cost = np.inf
        total = 0
        all_greedy_solns = list()
        start_time = time.time()
        for i in range(self.n):
            total += 1
            # print(f"starting from {i}")
            if time.time() - start_time > time_allowance:
                break

            route = self.greedy_path(i)
            if not route:
                continue
            cost = self.get_cost(route)
            if cost < np.inf:
                all_greedy_solns.append(route)
            if cost < bssf_cost:
                bssf_cost = cost
                bssf = route
                count += 1

        end_time = time.time()
        results["cost"] = bssf_cost
        results["time"] = end_time - start_time
        results["count"] = count
        results["soln"] = bssf
        results["max"] = None
        results["total"] = total
        results["pruned"] = None
        results["all"] = all_greedy_solns
        return results

    def branch_and_bound(self, time_allowance=60.0):
        results = {}
        start = time.time()
        print("\ngetting greedy results")
        initial_results = self.greedy(time_allowance)
        bssf_cost = initial_results["cost"]
        count = initial_results["count"]
        total = initial_results["total"]
        if not bssf_cost < np.inf:
            initial_results = self.defaultRandomTour(
                time_allowance - (time.time() - start)
            )
            count += initial_results["count"]
            total += initial_results["total"]
            bssf_cost = initial_results["cost"]

        bssf = initial_results["soln"]

        print(f"    initial solution found with cost: {timedelta(seconds=bssf_cost)} \t(T+{time.time()-start:.5f})")

        print("\nbeginning branch and bound")

        heap = []
        heapq.heappush(heap, TSPNode(self.matrix))
        max_heap_size = 0
        pop_pruned = 0
        push_pruned = 0
        total += 1

        max_possible_size = math.factorial(self.n)

        print_delta = 0.1
        last_print_time = start
        iteration = 0

        while len(heap) > 0 and time.time() - start < time_allowance:
            if time.time() - last_print_time > print_delta:
                last_print_time = time.time()
                print(f"(T+{time.time()-start:.5f}) considered: {total} pruned: {push_pruned+pop_pruned} heap size: {len(heap)}    ", end="\r")
            if len(heap) > max_heap_size:
                max_heap_size = len(heap)

            curr = heapq.heappop(heap)

            if curr.bound > bssf_cost:
                pop_pruned += 1
                continue

            for child in curr.children():
                total += 1
                soln, issoln = child.soln()
                if issoln and child.bound < np.inf:
                    cost = self.get_cost(soln)
                    if cost < bssf_cost:
                        print(
                            f"    found better solution with cost: {timedelta(seconds=cost)} (-{timedelta(seconds=bssf_cost-cost)})    (T+{time.time()-start:.5f})"
                        )
                        count += 1
                        bssf = soln
                        bssf_cost = cost
                elif child.bound < bssf_cost:
                    heapq.heappush(heap, child)
                else:
                    push_pruned += 1

        end = time.time()
        results["cost"] = bssf_cost
        results["time"] = end - start
        results["count_bssf"] = count
        results["soln"] = bssf
        results["max_heap_size"] = max_heap_size
        results["total_children_considered"] = total
        results["push_pruned"] = push_pruned
        results["pop_pruned"] = pop_pruned
        results["solved"] = len(heap) == 0
        results["heap"] = len(heap)
        return results


def get_addresses(filename="addresses.txt"):
    addresses = []
    with open(filename) as file:
        for line in file.readlines():
            addresses.append(line.strip())
    return addresses


def get_labels(filename="labels.txt"):
    labels = []
    with open(filename) as file:
        for line in file.readlines():
            labels.append(line.strip())
    return labels


def matrix_by_hand(addresses):
    if len(addresses) <= 10:
        return matrix_via_dist_mat(addresses)
    client = googlemaps.Client(API_KEY)
    time_matrix = np.zeros((len(addresses), len(addresses)))
    time_matrix[:10,:10] = matrix_via_dist_mat(addresses[:10])
    # print(time_matrix)
    for r in tqdm(range(len(addresses))):
        for c in tqdm(range(len(addresses)), leave=False, desc=addresses[r]):
            if r < 10 and c < 10:
                continue
            # print(r,addresses[r],c,addresses[c])
            maps_result = client.distance_matrix(addresses[r], addresses[c])
            time_matrix[r, c] = maps_result["rows"][0]["elements"][
                0
            ]["duration"]["value"]
    for i in range(len(time_matrix)):
        time_matrix[i, i] = np.inf
    return time_matrix


def matrix_via_dist_mat(addresses):
    if len(addresses) > 10:
        return matrix_by_hand(addresses)
    client = googlemaps.Client(key=API_KEY)
    matrix = client.distance_matrix(addresses, addresses)
    time_matrix = np.zeros((len(addresses), len(addresses)))
    r = 0
    for row in matrix["rows"]:
        c = 0
        for element in row["elements"]:
            # print(element['duration']['value'])
            time_matrix[r, c] = element["duration"]["value"]
            c += 1
        r += 1
    for i in range(len(time_matrix)):
        time_matrix[i, i] = np.inf
    return time_matrix


def get_distance_matrix(addresses):
    if len(addresses) > 10:
        print("too many waypoints, getting addresses by hand")
        return matrix_by_hand(addresses)
    return matrix_via_dist_mat(addresses)


def solve_TSP(addresses:list, time_allowance:float, start:str, opensolve:bool):
    """ Uses the TSPSolver class to solve the Travelling Salesman Problem. 

    Args:
        addresses (list): list of addresses to pass into the google maps API
        start (str): The starting address for fixed start, arbitrary end mode. If not found in addresses, will run in loop mode with the top address assumed to be the start.
        open (bool): Whether to run in arbitrary start, arbitrary end mode. Is overridden if start is valid.

    Returns:
        _type_: _description_
    """
    print("getting Google Maps Distances...")
    matrix = get_distance_matrix(addresses)

    # edit the matrix if necessary (for fixed start and open solve)
    start_idx = 0
    if start in addresses:
        start_idx = addresses.index(start)
        addresses.append('dummy')
        rows, cols = matrix.shape
        new_matrix = np.zeros((rows+1, cols+1))
        new_matrix[:rows,:cols] = matrix
        new_matrix[:,-1] = 0
        new_matrix[-1,:] = np.inf
        new_matrix[-1,start_idx] = 0
        matrix = new_matrix
    elif opensolve:
        addresses.append('dummy')
        rows, cols = matrix.shape
        new_matrix = np.zeros((rows+1, cols+1))
        new_matrix[:rows,:cols] = matrix
        new_matrix[-1,:] = 0
        new_matrix[:,-1] = 0
        new_matrix[-1,-1] = np.inf
        matrix = new_matrix
    # print(matrix)


    # print(matrix)
    solver = TSPSolver(matrix)
    results = solver.branch_and_bound(time_allowance)
    print("\n\n\n", json.dumps(results,indent=1),end="\n\n\n")
    cost = results["cost"]
    soln = results["soln"]
    comp_time = results["time"]
    print("\n")
    print(f"(T+{comp_time:.5f})")
    if not results["solved"]:
        print("Time ceiling hit. Results not guaranteed to be optimal")
    print(f"Solution found with cost: {timedelta(seconds=cost)}")

    

    # [print(addresses[i]) for i in soln]
    return [addresses[i] for i in soln], soln

def create_apple_maps_route(origin, destination, stops=None, mode="driving"):
    """Generates and opens an accurate Apple Maps route using universal parameters.

    - origin: String (Starting address)
    - destination: String (Final destination)
    - stops: List of strings (Optional intermediate addresses)
    - mode: 'driving', 'walking', or 'transit'
    """
    # Initialize the base query parameters dictionary
    params = {"source": origin}

    # Construct the base URL
    base_url = "https://maps.apple.com/directions"

    # Convert the base parameters dictionary to a URL-encoded string.
    # urlencode natively formats spaces as '+' rather than '%20'.
    query_string = f"source={urllib.parse.quote_plus(origin)}"

    # If intermediate stops are provided, append them cleanly to the URL string.
    # Note: Multi-stop paths are fully supported under driving mode.
    if isinstance(stops, list):
        for stop in stops:
            # Safely convert individual stop spaces to '+' format
            encoded_stop = urllib.parse.quote_plus(stop)
            query_string += f"&waypoint={encoded_stop}"
    query_string += f"&destination={urllib.parse.quote_plus(destination)}"

    # Assemble the final URL string
    maps_url = f"{base_url}?{query_string}&mode={mode}"

    print(f"Generated URL:\n{maps_url}\n")



if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    parser = argparse.ArgumentParser(
        prog="Travelling Salesman",
        description="Solve a traveling salesman problem. Will prompt for a time constraint. By default (no arguments), solve a closed-loop TSP. The starting node in this case is assumed to be the first in the address list."
    )
    parser.add_argument("time_allowance", default="60", help="the time allowance in seconds (not counting the time it takes to get the distance matrix)")
    parser.add_argument("--start", default="", help="the starting node label in fixed start, arbitrary end mode. Will search the labels list for a label string that begins with START. If no such labels are found, revert to the default behavior (or --open if applicable). If more than one matching label is found, use the first one in the list.")
    parser.add_argument("--open", action="store_true", help="whether to run in arbitrary start, arbitrary end mode. is overridden by --start")
    args = parser.parse_args()

    start, opensolve = args.start, args.open
    time_allowance = float(args.time_allowance)
    addresses = get_addresses()

    if not addresses:
        print(f"no addresses found!")
        exit()

    try:
        labels = get_labels()
    except:
        labels=[]
    matching_labels = [l for l in labels if l.lower().startswith(start.lower())] if start else[]
    if matching_labels:
        starting_label = matching_labels[0]
        starting_address = addresses[labels.index(starting_label)]
    else:
        matching_labels = [l for l in addresses if l.lower().startswith(start.lower())] if start else[]
        if matching_labels:
            starting_address = matching_labels[0]
        else:
            starting_address = ""
    if starting_address:
        print(f"starting from {starting_address}")
    



    
    route, soln = solve_TSP(addresses, time_allowance, starting_address, opensolve)
    if opensolve:
        if 'dummy' not in route:
            raise RuntimeError("Unable to find 'dummy' in route!")
        starting_index = (route.index('dummy') + 1)%len(route)
        starting_address = route[starting_index]
    if not starting_address: starting_address = addresses[0]


    # for i,a in enumerate(addresses):
    #     label = labels[i] if len(labels) > i else ""
    #     print(i,label, a)

    max_len = 0
    for l in labels:
        if len(l) > max_len:
            max_len = len(l)

    starting_index = route.index(starting_address)

    i = starting_index

    addresses_in_order = []

    while 1:
        index = soln[i]
        address = addresses[index]
        route_address = route[i]
        i = (i+1)%len(soln)

        if address != 'dummy':
            label=labels[index] if len(labels) > index else ""
            print(label.ljust(max_len + 1) + ":\t" + repr(address))

            addresses_in_order.append(address)

        if i == starting_index:
            break

    origin = addresses_in_order[0]
    addresses_in_order = addresses_in_order[1:]
    if start or opensolve:
        destination = addresses_in_order[-1]
        addresses_in_order = addresses_in_order[:-1]
    else:
        destination = origin
    create_apple_maps_route(origin=origin, stops=addresses_in_order, destination=destination)


    if not start and not opensolve:
        print("\nreversed:")
        addresses_in_order = []
        while 1:
            index = soln[i]
            address = addresses[index]
            route_address = route[i]
            i = (i-1)%len(soln)

            if address != 'dummy':
                label=labels[index] if len(labels) > index else ""
                print(label.ljust(max_len + 1) + ":\t" + repr(address))

                addresses_in_order.append(address)
            
            if i == starting_index:
                break
        origin = addresses_in_order[0]
        destination = origin
        addresses_in_order = addresses_in_order[1:]
        create_apple_maps_route(origin=origin, stops=addresses_in_order, destination=destination)

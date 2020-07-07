# distutils: language=c++

import math
import sys
import random
import statistics
import multiprocessing
import cProfile

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np


LINE_PROFILER_FLAG = True

PROFILER_FLAG = False
if PROFILER_FLAG:
    profiler = cProfile.Profile()

#Input graph in adjacency list format
adj_list = {}

filename = "ca-GrQc.txt" if len(sys.argv) < 2 else sys.argv[1]

with open(filename, 'r') as f:
    for line in f:
        line = line.strip()
        u, v = line.split()

        if int(u) == int(v):
            continue

        if int(u) not in adj_list:
            adj_list[int(u)] = {int(v): 1}
        else:
            if int(v) not in adj_list[int(u)]:
                adj_list[int(u)][int(v)] = 1

        if int(v) not in adj_list:
            adj_list[int(v)] = {int(u): 1}
        else:
            if int(u) not in adj_list[int(v)]:
                adj_list[int(v)][int(u)] = 1

#Sample list: adj_list = {3: [1, 2, 4], 1: [2, 3, 4], 2: [1, 3, 4], 4: [1, 3, 2, 5], 5: [4, 6], 6:[5]}

#Computes number of edges in the graph
M = sum(len(v) for ke, v in adj_list.items()) / 2

#Computes number of nodes in the graph
N = len(adj_list)

print("Processing graph with {} nodes and {} edges".format(N, M))

#nodes
nodes = np.array(list(adj_list.keys()))

delta = 0.5

S = M ** delta * math.log(N)

#k needs to be less than S/log n
k = max(2, int(6.5 * math.log(N)))

#k-wise independent hash function family
P = sys.hash_info.modulus
P = 2 ** 31 - 1
def kIndepHash(x, hash_coefficients):
    #Computing this takes too long.
    hash_value = 0
    x_product = 1
    for i in range(k):
       hash_value = (hash_value + (x_product * hash_coefficients[i])) % P
       x_product = (x_product * x) % P

    return hash_value / (P - 1)

# Machine IDs for each machine
machines = [x for x in range(math.ceil((M * math.log(M))/S))]

#Probability for sampling into a machine
coeff = 1.0/5
prob_sample = coeff * math.sqrt(S/(M * k))

#Constant factor in front of machine space
s = 2

# Assign nodes to machines
# @profile
def machineNodes():
    #keeps track of nodes assigned to machines
    global machines
    global nodes
    global prob_sample

    machine_nodes = {}
    factor = 10**(math.ceil(math.log(len(machines), 10)))

    hash_coefficients = np.random.randint(P, size=k)

    for machine in machines:
        cur_nodes = []

        # Use pseudorandom function in place of k-wise independent
        # hash function.
        random_hash_values = np.random.rand(N)

        current_machine_nodes = []
        for i in range(N):
            if random_hash_values[i] <= prob_sample:
                current_machine_nodes.append(nodes[i])
        machine_nodes[machine] = current_machine_nodes

    return machine_nodes

#machine nodes mapped to edges
def machineEdges(machine_nodes):
    global machines
    global adj_list
    global s
    global S

    machine_edges = {}

    for machine in machines:
        if machine in machine_nodes:
            mn = machine_nodes[machine]
        else:
            continue
        for i in mn:
            for j in mn:
                if i in adj_list and j in adj_list[i]:
                    if not machine in machine_edges:
                        machine_edges[machine] = {}
                    edges = machine_edges[machine]
                    if not (i, j) in edges and not (j, i) in edges:
                        machine_edges[machine][(i, j)] = 1
        if machine in machine_edges and len(machine_edges[machine]) > s * S:
            machine_edges.pop(machine)

    return machine_edges

#Power for the probability estimation
estimation_pow = 3

#Count triangles within each machine
def countTriangles(machine_edges):
    global machines
    global prob_sample
    global estimation_pow

    triangles = 0
    R = len(machine_edges)
    for machine in machines:
        if machine in machine_edges:
            edges = machine_edges[machine]
            for k1 in edges:
                for k2 in edges:
                    if not k1 == k2:
                        (u, v) = k1
                        (w, x) = k2

                        if u == w:
                            if (v, x) in edges or (x, v) in edges:
                                triangles += 1
                        elif u == x:
                            if (v, w) in edges or (w, v) in edges:
                                triangles += 1
                        elif v == w:
                            if (u, x) in edges or (x, u) in edges:
                                triangles += 1
                        elif v == x:
                            if (u, w) in edges or (w, u) in edges:
                                triangles += 1
    if R >= 1:
        return (1/(prob_sample**estimation_pow * R)) * (triangles/6)
    else:
        return 0

def countPartitionTriangles(machine_edges):
    global machines
    global prob_sample
    global estimation_pow

    triangles = 0
    R = len(machine_edges)
    for machine in machines:
        if machine in machine_edges:
            edges = machine_edges[machine]
            for k1 in edges:
                for k2 in edges:
                    if not k1 == k2:
                        (u, v) = k1
                        (w, x) = k2

                        if u == w:
                            if (v, x) in edges or (x, v) in edges:
                                triangles += 1
                        elif u == x:
                            if (v, w) in edges or (w, v) in edges:
                                triangles += 1
                        elif v == w:
                            if (u, x) in edges or (x, u) in edges:
                                triangles += 1
                        elif v == x:
                            if (u, w) in edges or (w, u) in edges:
                                triangles += 1
    p = 1.0/len(machines)
    return (1/p**2) * triangles

def computeApproxTriangleCount(N):
    if PROFILER_FLAG:
        profiler.enable()
    machine_nodes = machineNodes()
    if PROFILER_FLAG:
        profiler.disable()
    machine_edges = machineEdges(machine_nodes)
    ct = countTriangles(machine_edges)

    return ct

def main():
    global N

    approx = []

    multiplier = 100
    num_iterations = int(multiplier * math.ceil(math.log(N)))
    progressbar_width = 100
    progresbar_update_interval = num_iterations // progressbar_width

    thread_count = multiprocessing.cpu_count()
    #print("Running {} iterations in parallel on {} threads".format(num_iterations, thread_count))

    # setup progressbar
    sys.stdout.write("[%s]" % (" " * progressbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (progressbar_width + 1)) # return to start of line, after '['

    if PROFILER_FLAG or LINE_PROFILER_FLAG:
        for i in tqdm(range(num_iterations)):
            approx.append(computeApproxTriangleCount(N))
    else:
        input_array = [N for _ in range(num_iterations)]
        with ProcessPoolExecutor(max_workers=thread_count) as executor:
            for r in tqdm(executor.map(computeApproxTriangleCount, input_array), total=num_iterations):
                approx.append(r)

    if PROFILER_FLAG:
        profiler.print_stats(sort="time")

    print("\n")
    approx.sort()

    return statistics.median(approx)

#Partition Algorithm
def partitionAlgorithm(N):
    global nodes
    global M
    global adj_list
    global machines

    C = int(len(machines))

    node_to_color = {}

    for n in nodes:
        machine = random.randint(0, C-1)
        if machine not in node_to_color:
            node_to_color[machine] = {}
        node_to_color[machine][n] = 1

    machine_edges = machineEdges(node_to_color)
    tris = countPartitionTriangles(machine_edges)
    return tris

def partitionMain():
    global N

    approx = []

    multiplier = 100
    num_iterations = int(multiplier * math.ceil(math.log(N)))
    progressbar_width = 100
    progresbar_update_interval = num_iterations // progressbar_width

    thread_count = multiprocessing.cpu_count()
    print("Running {} iterations in parallel on {} threads".format(num_iterations, thread_count))

    # setup progressbar
    sys.stdout.write("[%s]" % (" " * progressbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (progressbar_width + 1)) # return to start of line, after '['

    if PROFILER_FLAG or LINE_PROFILER_FLAG:
        for i in tqdm(range(num_iterations)):
            approx.append(partitionAlgorithm(N))
    else:
        input_array = [N for _ in range(num_iterations)]
        with ProcessPoolExecutor(max_workers=thread_count) as executor:
            for r in tqdm(executor.map(partitionAlgorithm, input_array), total=num_iterations):
                approx.append(r)

    if PROFILER_FLAG:
        profiler.print_stats(sort="time")
    print("\n")
    approx.sort()

    return statistics.median(approx)

def exactCount(adj_list):
    triangles = 0

    edges = {}
    for key in adj_list:
        for neighbor in adj_list[key]:
            if (key, neighbor) not in edges and (neighbor, key) not in edges:
                edges[(key, neighbor)] = 1

    for k1 in edges:
        for k2 in edges:
            if not k1 == k2:
                (u, v) = k1
                (w, x) = k2

                if u == w:
                    if (v, x) in edges or (x, v) in edges:
                        triangles += 1
                elif u == x:
                    if (v, w) in edges or (w, v) in edges:
                        triangles += 1
                elif v == w:
                    if (u, x) in edges or (x, u) in edges:
                        triangles += 1
                elif v == x:
                    if (u, w) in edges or (w, u) in edges:
                        triangles += 1

    return triangles/6

# Facebook graph exact number of triangles
#trian = 1612010.0

# Amazon graph exact number of triangles
#trian = 667129

#LocBright graph exact number of triangles
#trian = 494728

#musae de
#trian = 603088

#hep-ph
#trian = 3358499

#gr-cp
#trian = 48260

#hep-th
#trian = 28339

#last-fm
trian = 40433

print("Total space, machine space, delta, probability: {} {} {} {}".format(M * math.log(M), s * S, delta, prob_sample))
approx = main()
print("Exact count: " + str(trian))
print("Our Approx count: " + str(approx))
partition = partitionMain()
print("Partition algorithm count: {}".format(partition))
print("Partition approx: " + str(partition/trian))

if trian > 0:
    print("Our Approx ratio: " + str(approx/trian))
else:
    print("No triangles.")

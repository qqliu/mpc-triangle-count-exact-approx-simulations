import math
import sys
import random
import statistics
import copy
import cProfile
import multiprocessing

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

PROFILER = False
fastMethod = False
slowMethod = False
sequential = True

if PROFILER:
    profiler = cProfile.Profile()

f = 'facebook_combined.txt'

files = ['facebook_combined.txt', 'ca-GrQc.txt', 'oregon1_010519.txt', 'ca-HepTh.txt', 'email-Enron.txt', 'ca-CondMat.txt', 'ca-HepPh.txt',
        'com-amazon.ungraph.txt', 'facebook_combined.txt', 'loc-brightkite_edges.txt']

arbs = [20, 34]

deltas = [0.3, 0.5, 0.9]

for ar in arbs:
    for de in deltas:
        args = ((f, a, d) for a in arbs for d in deltas)

#MPC Sorting algorithm
def MPCSort(items, machines, S):
    #Rounds spent on sorting
    R = 0

    n = len(items)
    num_machines = len(machines)

    if n > int(S):
        pivots = random.choices(items, k = min(int(S), num_machines))
    else:
        pivots = items
        pivots.sort()
        return pivots, 0

    R += 1

    # Sort the pivots
    pivots.sort()

    machines_to_items = {}
    for m in machines:
        machines_to_items[m] = []

    pl = len(pivots)

    #Put items into buckets
    for item in items:
        for i in range(pl):
            if i == 0:
                if item < pivots[i]:
                    machines_to_items[i].append(item)
                    break
            if not i == pl - 1:
                if pivots[i] == pivots[i+1]:
                    if len(machines_to_items[i]) < int(S) and item == pivots[i]:
                        machines_to_items[i].append(item)
                        break
                    else:
                        continue
                else:
                    if item >= pivots[i] and item < pivots[i+1]:
                        machines_to_items[i].append(item)
                        break
            else:
                if item >= pivots[pl - 1]:
                    machines_to_items[i].append(item)
                    break

    # One round of communication necesary to put items into buckets
    R += 1

    #Sort the buckets
    sortedItems = []

    #Maximum number of rounds necesarily to sort all the buckets in parallel
    maxr = 0

    #If all the items can be sorted in one bucket, sort it
    #Otherwise, recursively call this method
    for m in machines:
        ml = len(machines_to_items[m])
        if ml > S:
            newlist = sorted(machines_to_items[m])
            if newlist[0] == newlist[ml - 1]:
                sortedItems += newlist
            else:
                it, r = MPCSort(machines_to_items[m], machines, S)
                r = 0
                sortedItems += it

                if r > maxr:
                    maxr = r
        elif ml > 0:
            machines_to_items[m].sort()
            sortedItems += machines_to_items[m]
    return sortedItems, R + maxr

#MPC Find Duplications algorithm
def MPCFindDuplicates(items, machines, S):
    R = 0
    leaves = {}

    #Put items into "leaf" machines
    for m in machines:
        leaves[m] = []

    empty = 0
    for item in items:
        if len(leaves[empty]) < int(S):
            leaves[empty].append(item)
        else:
            empty += 1
            leaves[empty].append(item)

    #Remove empty machines at the leaves
    remove = []
    for m in leaves:
        if len(leaves[m]) == 0:
            remove.append(m)

    for m in remove:
        leaves.pop(m)

    # Count the duplicates in each machine at the leaves
    node_machine_counts = {}

    for m in leaves:
        node_machine_counts[m] = []

    for m in leaves:
        prev = None
        count = 0
        for n in leaves[m]:
            if prev == None:
                prev = n
            if n == prev:
                count += 1
            else:
                node_machine_counts[m].append((prev, count))
                prev = n
                count = 1
        node_machine_counts[m].append((prev, count))

    # Remove empty leaves if necessary
    remove = []
    for m in node_machine_counts:
        if len(node_machine_counts[m]) == 0:
            remove.append(m)

    for m in remove:
        node_machine_counts.pop(m)

    tree_nodes = len(node_machine_counts)
    levels = {}

    # Compute duplicates across all machines by sending boundaries of machines up to its parent machines
    # By Theorem 2.1
    c = 0
    while tree_nodes > 1:
        R += 1

        levels[c] = {}

        for m in range(tree_nodes):
            levels[c][m] = []

        curLevel = 0

        if c == 0:
            for m in node_machine_counts:
                if len(levels[c][curLevel]) >= int(S):
                    curLevel += 1

                mnl = len(node_machine_counts[m])
                if mnl == 1:
                    levels[c][curLevel].append(node_machine_counts[m][0])
                else:
                    levels[c][curLevel].append(node_machine_counts[m][0])
                    levels[c][curLevel].append(node_machine_counts[m][mnl-1])

        else:
            for m in levels[c-1]:
                prev = None
                new_pairs = []
                for p in levels[c-1][m]:
                    if prev == None:
                        prev = p
                    else:
                        if prev[0] == p[0]:
                            prev = (prev[0], prev[1] + p[1])
                        else:
                            new_pairs.append(prev)
                            prev = p
                new_pairs.append(prev)

                if len(levels[c][curLevel]) >= int(S):
                    curLevel += 1

                npl = len(new_pairs)
                if npl == 1:
                    levels[c][curLevel].append(new_pairs[0])
                else:
                    levels[c][curLevel].append(new_pairs[0])
                    levels[c][curLevel].append(new_pairs[npl - 1])
                levels[c-1][m] = new_pairs

        remove = []
        for m in levels[c]:
            if len(levels[c][m]) == 0:
                remove.append(m)

        for m in remove:
            levels[c].pop(m)

        tree_nodes = len(levels[c])
        c += 1

    prev = None
    new_pairs = []

    node_counts = {}
    if c > 0:
        c -= 1
        for m in levels[c]:
            for p in levels[c][m]:
                if prev == None:
                    prev = p
                else:
                    if prev[0] == p[0]:
                        prev = (p[0], prev[1] + p[1])
                    else:
                        new_pairs.append(prev)
                        prev = p
            new_pairs.append(prev)
            levels[c][m] = new_pairs

        # Send duplicates down to the leaves
        while c > -1:
            R += 1
            for m in levels[c]:
                for p in levels[c][m]:
                    if not p[0] in node_counts:
                        node_counts[p[0]] = p[1]
            c -= 1

    # Count any nodes in the leaves
    for m in node_machine_counts:
        for p in node_machine_counts[m]:
            if not p[0] in node_counts:
                node_counts[p[0]] = p[1]
    return node_counts, R

#Find triangles adjacent to specific neighbors
def findTriangleAdj(A, adj_list, deg, w, machines, S):
    items = []
    for a in A:
        if a in adj_list[w]:
            items += adj_list[a]
    items += adj_list[w]
    if len(items) > math.ceil(S):
        sortedNeighbors, R = MPCSort(items, machines, S)
    else:
        sortedNeighbors = sorted(items)
        R = 0
    if len(sortedNeighbors) > math.ceil(S):
        node_counts, r = MPCFindDuplicates(sortedNeighbors, machines, S)
    else:
        node_counts = {}
        for y in sortedNeighbors:
            if not y in node_counts:
                node_counts[y] = 1
            else:
                node_counts[y] += 1
        r = 0

    T = 0
    for v in adj_list[w]:
        vl = len(adj_list[v])
        wl = len(adj_list[w])
        if vl > deg and wl > deg:
            T += 6 * (node_counts[v] - 1)
        elif (vl > deg and wl <= deg) or (vl <= deg and wl > deg):
            T += 3 * (node_counts[v] - 1)
        else:
            T += 2 * (node_counts[v] - 1)

    return T, R + r

#Find number of triangles adjacent to A using MPC methods
def findTriangles(A, deg, adj_list, machines, S):
    Tr = 0

    # setup progressbar
    progressbar_width = 100
    sys.stdout.write("[%s]" % (" " * progressbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (progressbar_width + 1)) # return to start of line, after '['

    # Imitate Algorithm 4: Count-Triangles
    maxR = 0
    for w in tqdm(adj_list):
        T, R = findTriangleAdj(A, adj_list, deg, w, machines, S)

        Tr += T
        if R > maxR:
            maxR = R

    return Tr, maxR

#Finding number of triangles given a set of nodes *sequentially*
def findSeqTriangles(A, adj_list):
    t = 0
    for a in A:
        neighbors = adj_list[a]
        for n1 in neighbors:
            for n2 in neighbors:
                if not n1 == n2:
                    if n2 in adj_list[n1]:
                        if not n1 in A and not n2 in A:
                            t += 6
                        elif n1 in A and not n2 in A:
                            t += 3
                        elif not n1 in A and n2 in A:
                            t += 3
                        else:
                            t += 2
    t = t/12
    return t, 1

#Count triangles procedure
def countTriangles(adj_list, machines, S, N, arb):
    #keep track of the current degrees of the nodes
    Q = {}

    # Counts triangles
    T = 0

    #Counts number of rounds
    R = 0

    # Compute degrees in induced subgraph that's left
    for node in adj_list:
        Q[node] = len(adj_list[node])

    # maximum number of rounds needed
    num_rounds = 2*math.ceil(math.log(math.log(N, 2), 3.0/2))

    for i in range(num_rounds):
        #degree threshold
        deg = 2**((3.0/2)**i) * (2 * arb)

        #list of vertices to count that has degree less than deg
        A = {}
        for node in Q:
            if Q[node] <= deg:
                A[node] = 1

        if not A and Q:
            return "Arboricity is too small", ""

        t, r = findTriangles(A, deg, adj_list, machines, S)
        T += t
        R += r

        for node in A:
            for neighbor in adj_list[node]:
                if neighbor in adj_list and node in adj_list[neighbor]:
                    adj_list[neighbor].pop(node)

            adj_list.pop(node)

        #reset Q
        Q = {}
        for node in adj_list:
            Q[node] = len(adj_list[node])

        if not Q:
            R += i + 1
            break

        #This happens if degeneracy estimate is too small
        if R == 0:
            R = -1

    return T/12, R

#Count triangles sequentially procedure (used for checking triangle counts are correct)
def countSeqTriangles(adj_list, arb, N):
    #keep track of the current degrees of the nodes
    Q = {}

    # Counts triangles
    T = 0

    #Counts number of rounds
    R = 0

    for node in adj_list:
        Q[node] = len(adj_list[node])

    # number of rounds
    num_rounds = 2*math.ceil(math.log(math.log(N, 2), 3.0/2))

    for i in range(num_rounds):
        #degree threshold
        deg = 2**((3.0/2)**i) * (2 * arb)

        #list of vertices to count
        A = {}
        for node in Q:
            if Q[node] <= deg:
                A[node] = 1

        if not A and Q:
            return "Arboricity is too small", ""

        t, r = findSeqTriangles(A, adj_list)
        T += t
        R += r

        for node in A:
            for neighbor in adj_list[node]:
                if neighbor in adj_list and node in adj_list[neighbor]:
                    adj_list[neighbor].pop(node)

            adj_list.pop(node)

        #reset Q
        Q = {}
        for node in adj_list:
            Q[node] = len(adj_list[node])

        if len(Q.keys()) == 0:
            R += i + 1
            break
    return T, R

#Method for computing the degeneracy
def computeArboricity(adj_list, N):
    max_min_deg = 0

    while len(adj_list) > 0:
        min_deg = N
        for node in adj_list:
            if len(adj_list[node]) < min_deg:
                min_deg = len(adj_list[node])

        remove = {}
        for node in adj_list:
            if len(adj_list[node]) <= min_deg:
                remove[node] = 1

        for node in remove:
            for neighbor in adj_list[node]:
                adj_list[neighbor].pop(node)
            adj_list.pop(node)
        if min_deg > max_min_deg:
            max_min_deg = min_deg
    return max_min_deg

#Count triangles procedure ONLY using arboricty as degree constraint
def countTrianglesArb(adj_list, adj_2, M, S, N):
    #keep track of the current degrees of the nodes
    Q = {}

    # Counts triangles
    T = 0

    #Counts number of rounds
    R = 0

    #Total space
    factor = 5
    realArb = computeArboricity(adj_2, N)
    TS = factor * realArb * M

    #Machines computed from total space and IDs
    machines = [x for x in range(math.ceil(TS/S))]

    # Compute degrees in induced subgraph that's left
    for node in adj_list:
        Q[node] = len(adj_list[node])

    # maximum number of rounds needed
    num_rounds = 2*math.ceil(math.log(N, 2))

    for i in range(num_rounds):
        #degree threshold
        deg = realArb

        #list of vertices to count that has degree less than deg
        A = {}
        for node in Q:
            if Q[node] <= deg:
                A[node] = 1

        if not A and Q:
            return "Arboricity is too small", ""

        t, r = findTriangles(A, deg, adj_list, machines, S)
        T += t
        R += r

        for node in A:
            for neighbor in adj_list[node]:
                if neighbor in adj_list and node in adj_list[neighbor]:
                    adj_list[neighbor].pop(node)

            adj_list.pop(node)

        #reset Q
        Q = {}
        for node in adj_list:
            Q[node] = len(adj_list[node])

        if not Q:
            R += i + 1
            break

        #This happens if degeneracy estimate is too small
        if R == 0:
            R = -1

    return T/12, R, realArb

def main(adj_list, machines, S, N, arb):
    return countTriangles(adj_list, machines, S, N, arb)

def run(filename, ar, de):
    #Input graph in adjacency list format
    adj_list = {}

    # Compute edges from edges file
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            u, v = line.split()

            if int(u) == int(v):
                continue

            if int(u) not in adj_list:
                adj_list[int(u)] = {int(v): 1}
            else:
                adj_list[int(u)][int(v)] = 1

            if int(v) not in adj_list:
                adj_list[int(v)] = {int(u): 1}
            else:
                adj_list[int(v)][int(u)] = 1

    #Computes number of edges in the graph
    M = sum(len(v) for ke, v in adj_list.items()) / 2

    #Computes number of nodes in the graph
    N = len(adj_list)

    #Guess arboricity of the graph
    arb = ar

    #Space per machine exponent
    delta = de

    #Space per machine
    S = N ** delta

    #Constant factor in front of total space
    factor = 5

    #Total space
    TS = factor * arb * M

    #Machines computed from total space and IDs
    machines = [x for x in range(math.ceil(TS/S))]

    adj_2 = copy.deepcopy(adj_list)
    adj_3 = copy.deepcopy(adj_list)

    if PROFILER:
        profiler.enable()

    if PROFILER:
        profiler.disable()
        profiler.print_stats(sort="time")

    if slowMethod:
        TA, RA, realArb = countTrianglesArb(adj_2, adj_3, M, S, N)
        print("{} {} {} {} {} {}".format(filename, TA, RA, realArb, delta, int(N**delta)))

    if fastMethod:
        T, R = main(adj_list, machines, S, N, arb)
        print("{} {} {} {} {} {}".format(filename, T, R, arb, delta, int(N**delta)))

    if sequential:
        print(countSeqTriangles(adj_list, 5, N))

for arg in args:
    run(arg[0], arg[1], arg[2])

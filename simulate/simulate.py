
import sys
import numpy as np
import pandas as pd
import networkx as nx
from collections import deque
import math

from collections import deque
import random

from shapely.geometry import LineString


def by_distance(element):
    return element[1]


def get_closest(G, locations, node, q):
    # get position of node
    x = locations[locations['ID'] == _id[node]]['Longitude'].iloc[0]
    y = locations[locations['ID'] == _id[node]]['Latitude'].iloc[0]
    
    #Neighbors
    ng = list(G.neighbors(node))
    
    # Calculate distances
    distances = list()
    for n in ng:
        n_id = _id[n]
        x_row = locations[locations['ID'] == n_id]['Longitude'].iloc[0]
        y_row = locations[locations['ID'] == n_id]['Latitude'].iloc[0]
        d = (x - x_row)**2 + (y - y_row)**2
        distances.append((n, d))
    
    # Order by distance and return q closest
    distances.sort(key = by_distance)    
    return [i[0] for i in distances[:q]] 


def get_random_tree(G, nodes, WTP=None):
    new_G = nx.DiGraph()
    N = len(G.nodes())

    if WTP is None:
        WTP = random.choice(list(G.nodes()))

    V = [0] * N;  V[WTP] = 1
    Q = deque([]); Q.append(WTP)
    while Q:
        if len(new_G.nodes()) < nodes:
            random.shuffle(Q)
            u = Q.popleft()
            for v in G.neighbors(u):
                if not V[v]:
                    Q.append(v)
                    new_G.add_edge(v,u)
                    V[v] = 1
        else:
            break
    return new_G, WTP


def add_extra_edges(T, G, locations, ratio, edge_limit):
    nodes = []
    for u in T.nodes():
        if u in G.nodes():
            nodes.append(u)
    counter = 0
    perturbed = random.choices(nodes, k = math.ceil(len(nodes) * ratio))
    for node in perturbed:
        added = False
        if len(list(T.predecessors(node))) < edge_limit:
            closest = [i for i in get_closest(G, locations, node, 7) if i in nodes]
            if closest:
                closest = closest[0]
                T_neigbors = list(T.predecessors(node)) + list(T.successors(node))
                if len(list(T.predecessors(closest))) < edge_limit and  closest not in T_neigbors:
                    T.add_edge(node, closest)
                    added = True
                    #print('added')
        if not added and counter < len(nodes):
            perturbed.append(random.choice(nodes))
            counter += 1
    return T


def get_ideal(G, V_, u):
    
    if V_[u]:
        return list()
    
    V = V_.copy()
    s = [u];  V[u] = 1
    Q = deque([]); Q.append(u)
    while Q:
        u = Q.popleft()
        for v in G.predecessors(u):
            if not V[v]:
                s.append(v)
                V[v] = 1; Q.append(v)
    
    return s


def visit(G, V, u):
    
    if V[u]:
        return V, 0
    
    V[u] = 1; ans = 0
    Q = deque([]); Q.append(u)
    while Q:
        u = Q.popleft(); ans += 1
        for v in G.predecessors(u):
            if not V[v]:
                V[v] = 1
                Q.append(v)
    
    return V, ans


def get_size(G, V_, u):
    
    if V_[u]:
        return 0
    
    V = V_.copy()
    
    ans = 1;  V[u] = 1
    Q = deque([]); Q.append(u)
    while Q:
        u = Q.popleft()
        for v in G.predecessors(u):
            if not V[v]:
                ans += 1; V[v] = 1
                Q.append(v)
    
    return ans


def get_ideal_robust(G, V_, root, node):
    
    if root == node:
        return get_ideal(G, V_, root)
    
    V = [0 for u in range(len(V_))]
    V[root] = 1
    Q = deque([root])
    while Q:
        u = Q.popleft()
        for v in G.predecessors(u):
            if not V[v] and v != node:
                V[v] = 1
                Q.append(v)

    return [i for i in G.nodes() if (V_[i] == 0 and V[i] == 0)]


def visit_robust(G, V, root, u):
    
    I = get_ideal_robust(G, V, root, u)
    
    V_ = V.copy()
    for u in I:
        V_[u] = 1
    
    return V_, len(I)


def get_size_robust(G, V_, root, node):
    
    N = len(V_)
    
    if V_[node]:
        return 0
    
    if node == root:
        return N - sum(V_)
    
    V = [0] * N
    
    V[root] = 1; Q = deque([root])
    while Q:
        u = Q.popleft()
        for v in G.predecessors(u):
            if not V[v] and v != node:
                V[v] = 1
                Q.append(v)

    return sum([1 for u in range(N) if V_[u] + V[u] == 0])


def get_size_weight(G, W, V_, u):
    
    if V_[u]:
        return 0, 0
    
    V = V_.copy()
    
    s = 1; w = W[u]; V[u] = 1
    Q = deque([]); Q.append(u)
    while Q:
        u = Q.popleft()
        for v in G.predecessors(u):
            if not V[v]:
                s += 1; w += W[v]; V[v] = 1
                Q.append(v)
    
    return s, w


def get_size_weight_robust(G, W, V_, root, node):
    N = len(G.nodes())
    
    if V_[node]:
        return 0 , 0
    
    if node == root:
        return N - sum(V_), sum(W) - sum([W[i] for i in G.nodes() if V_[i]])
    
    V = [0] * len(V_)
    
    V[root] = 1; Q = deque([root])
    while Q:
        u = Q.popleft()
        for v in G.predecessors(u):
            if not V[v] and v != node:
                V[v] = 1
                Q.append(v)
                
    I = [i for i in G.nodes() if (V_[i] == 0 and V[i] == 0)]
    return sum([1 for i in I]), sum([W[i] for i in I])


def greedyAppReduceAll(G, W, V_, root, K, slim, wlim, plim, calcRobust, visitRobust):
    
    V = V_.copy(); N = len(G.nodes()); N_ = len(W)

    ans = []; tot = 0
    while len(ans) < K:
        
        E = []
        
        S = [0 for u in range(N_)]
        SS = [0 for u in range(N_)]
        for v in G.nodes():
            if not V[v]:
                ss = get_size(G, V, v)
                sr = get_size_robust(G, V, root, v)
                if calcRobust:
                    SS[v], S[v] = get_size_weight_robust(G, W, V, root, v)
                else:
                    SS[v], S[v] = get_size_weight(G, W, V, v)
                if (sr / ss) >= plim:
                    E.append(v)
        
        maxv = 0; u = -1
        for v in E:
            s = SS[v]
            if s > maxv and s <= slim and S[v] <= wlim:
                u = v
                maxv = s
            
        if u == -1:
            break
        
        x = None
        if visitRobust:
            V, x = visit_robust(G, V, root, u)
        else:
            V, x = visit(G, V, u)
        ans.append(u); tot += x
        
    return ans, tot


def simulate_robust_randtree(G, root, W, k2, S, plim, calcRobust, visitRobust, verbose, map_prev):
    
    iters = []
    
    N = len(G.nodes())
    N_ = len(W)

    node_it = 0
    for r in S:
        
        node_it += 1

        if verbose:
            print(r)

        CV = [0 for u in range(N_)]

        CV[r] = 1
        Q = deque([]); Q.append(r)
        while Q:
            u = Q.popleft()
            successors = list(G.successors(u))
            if successors:
                v = random.sample(successors, 1)[0]
                CV[v] = 1
                Q.append(v)

        V = [1 for u in range(N_)]
        for u in G.nodes():
            V[u] = 0

        R = N
        for t in range(100):
            
            R = N_ - sum(V); P = []; sP = 0
            
            if not verbose:
                print("                                                           ", end="\r")
                print(f"Search {r}: {t + 1}    now: {R}", end="\r")
            
            if verbose:
                print("it,", t, R)
                
            nn = 0
            for i in G.nodes():
                if V[i]:
                    nn += 2**i
            
            if nn in map_prev.keys():
                P = map_prev[nn]
            else:
                if R > N / 10:
                    P, sP = greedyAppReduceAll(G, W, V, root, k2, R / 3, 1e8, plim, calcRobust, visitRobust)
                elif R >= 10:
                    low = 0; high = R
                    while low != high:
                        mid = (low + high) // 2
                        P, sP = greedyAppReduceAll(G, W, V, root, k2, mid, 1e8, plim, calcRobust, visitRobust)
                        if R - sP < mid:
                            high = mid
                        else:
                            low = mid + 1
                    P, sP = greedyAppReduceAll(G, W, V, root, k2, low, 1e8, plim, calcRobust, visitRobust)
                    
                    if verbose:
                        print("pre low:", low, R - sP)

                    if low > 1:
                        P_, sP_ = greedyAppReduceAll(G, W, V, root, k2, low - 1, 1e8, plim, calcRobust, visitRobust)

                        if verbose:
                            print("low:", low, R - sP, R - sP_)

                        if abs((R - sP_) - (low - 1)) < abs((R - sP) - low) or (len(P) == 1 and sP == R):
                            P = P_; sP = sP_
                else:
                    P, sP = greedyAppReduceAll(G, W, V, root, k2, 1, 1e8, plim, calcRobust, visitRobust)

                map_prev[nn] = P
    
            if verbose:
                print("nx", len(P), sum([CV[u] for u in P]), N - sum(V))
                print("P: ", ' '.join([str(p) for p in P]))

            if sum([CV[u] for u in P]):
                V_ = [-1 * V[u] for u in range(N_)]
                for u in P:
                    if CV[u]:
                        I = get_ideal(G, V, u)
                        for v in I:
                            V_[v] += 1

                V = [(V_[u] != sum([CV[u] for u in P])) for u in range(N_)]
                
                if verbose:
                    print("if: ", N_ - sum(V))
            
            for u in P:
                if not CV[u]:
                    I = get_ideal_robust(G, V, root, u)
                    for v in I:
                        V[v] = 1
                        
            size = N_ - sum(V)
            weight = sum([W[u] for u in range(N_) if not V[u]])
            
            if verbose:
                print(size, weight)

            if size == 1 or weight <= 200:
                iters.append(t + 1)
                print(f"[{node_it}] Result {r}: {iters[-1]}    acc: {sum(iters) / len(iters)}")
                break
            if size <= k2:
                iters.append(t + 2)
                print(f"[{node_it}] Result {r}: {iters[-1]}    acc: {sum(iters) / len(iters)}")
                break
            if t == 99:
                iters.append(100)
                print(f"[{node_it}] Result {r}: {iters[-1]}    acc: {sum(iters) / len(iters)}")

    print(sum(iters) / len(iters), max(iters), "\n\n\n")
    
    return iters, map_prev


def get_location_node(locations, node):
    x = locations[locations['ID'] == node]['Longitude'].iloc[0]
    y = locations[locations['ID'] == node]['Latitude'].iloc[0]
    return x,y


def create_linestring_coord(x_1, y_1, x_2, y_2):
    return LineString([(x_1, y_1) , (x_2, y_2)])


def get_random_tree_divide(G, nodes, mxl, nodes_location, edges_location, WTP=None):

    new_edges = edges_location.copy()
    new_nodes = nodes_location.copy()

    last_edge_id = int(new_edges.tail(1)['edge_ID'])
    last_node_id = max(max(new_edges['ID_1']), max(new_edges['ID_2']))

    new_G = nx.DiGraph()
    N = len(G.nodes())

    if WTP is None:
        WTP = random.choice(list(G.nodes()))

    V = [0] * N;  V[WTP] = 1
    Q = deque([]); Q.append(WTP)
    while Q:
        if len(new_G.nodes()) < nodes:

            random.shuffle(Q)

            u = Q.popleft()
            xu, yu = get_location_node(nodes_location, u)

            for v in G.neighbors(u):
                if not V[v]:

                    Q.append(v); V[v] = 1
                    
                    xv, yv = get_location_node(nodes_location, v)
                    dist = np.sqrt((xu - xv) ** 2 + (yu - yv) ** 2)

                    xd, yd = (xu - xv) / dist, (yu - yv) / dist

                    while dist > mxl:

                        xvv, yvv = xv + xd * mxl, yv + yd * mxl
                        last_edge_id += 1
                        last_node_id += 1
                        new_G.add_edge(v, last_node_id)
                        new_edges = new_edges.append({'edge_ID': last_edge_id, 'ID_1': v, 'ID_2': last_node_id, 'Distance': mxl, 
                                                      'geometry': create_linestring_coord(xv, yv, xvv, yvv)}, ignore_index=True)
                        new_nodes = new_nodes.append({'ID': last_node_id, "Longitude": xvv, "Latitude": yvv}, ignore_index=True)

                        xv, yv = xvv, yvv
                        dist = np.sqrt((xu - xv) ** 2 + (yu - yv) ** 2)
                        v = last_node_id
                
                    last_edge_id += 1
                    new_G.add_edge(v, u)
                    new_edges = new_edges.append({'edge_ID': last_edge_id, 'ID_1': v, 'ID_2': u, 'Distance': dist, 
                                                  'geometry': create_linestring_coord(xv, yv, xu, yu)}, ignore_index=True)

        else:
            break

    return new_G, WTP, new_nodes, new_edges



if __name__ == "__main__":


    graph_size = int(sys.argv[1])
    K = int(sys.argv[2])
    root = int(sys.argv[3])
    ratio = float(sys.argv[4])
    its = int(sys.argv[5])
    output_file = sys.argv[6]

    path_nodes = 'TG.txt'
    nodes_location = pd.read_csv(path_nodes, sep=" ", header=None, names = ['ID', 'Longitude', 'Latitude'])
    path_edges = 'TG_edge.txt'
    edges_location = pd.read_csv(path_edges, sep=" ", header=None, names = ['edge_ID', 'ID_1', 'ID_2', 'Distance'])
    edges_location.head()

    S = set()
    for index, row in edges_location.iterrows():
        origin = row['ID_1']
        dest = row['ID_2']
        S.add(origin)
        S.add(dest)

    S = list(S)
    id_ = {}; _id = {}; l = 0
    for u in S:
        id_[u] = l; _id[l] = u
        l += 1

            
    G_SJ = nx.Graph()
    for index, row in edges_location.iterrows():
        origin = row['ID_1']
        dest = row['ID_2']
        G_SJ.add_edge(id_[origin], id_[dest])
        
    N_ = l



    dist_lim = 250

    for _ in range(0, its):

        print(f"\n\n\n{_ + 1} RUN \n\n\n")
        
        T, WTP, new_nodes, new_edges = get_random_tree_divide(G_SJ, graph_size, dist_lim, nodes_location, edges_location, root)

        N_ = max(T.nodes()) + 1
        r = WTP

        W = [100000 for u in range(N_)]

        iters, _ = simulate_robust_randtree(T, r, W, K, T.nodes(), 0, True, True, False, {})

        G = add_extra_edges(T, G_SJ, nodes_location, ratio, 5)

        N_ = max(G.nodes()) + 1

        iters, _ = simulate_robust_randtree(G, r, W, K, G.nodes(), 0, True, True, False, {})
        
        with open(output_file, "a") as myfile:
            myfile.write(' '.join([str(i) for i in iters]) + '\n')
        

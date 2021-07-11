# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)
import queue
import itertools
from math import *
from heapq import *
from time import process_time



def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)




# Reference: 
# Version: 0.2 / Edited: Hyun Do Jung
# Comments: 0.1 :: bfs calls bfs_solve (reucrsive call on bfs)
#           0.2 :: returned to recursive bfs calls for better point sets
# 

# Breadth-first search algorithm 
def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    q = queue.Queue()
    q.put(start)

    visited = {}

    predecessors = {}
    predecessors[start] = None

    while (not q.empty()):


        first = q.get()
        current_pred = first
        visited[first] = 1
        neighbors = maze.getNeighbors(first[0], first[1])

        for neighbor in neighbors:
            if (not neighbor in visited) and (maze.isValidMove(neighbor[0], neighbor[1])):
                predecessors[neighbor] = current_pred
                visited[neighbor] = 1
                q.put(neighbor)

    dots = maze.getObjectives()
    goal = maze.getObjectives()[0]
    path = []
    path.append(goal)
    current = goal

    while (current != start):
        path.append(predecessors[current])
        current = predecessors[current]

    path.append(start)
    print(path)
    return path

"""
def bfs_solve(maze, start, objectives, q, visited):
    
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on
           start: start - tuple
           objectives: list of objectives - tuple
           q: queue that contains start vertex and the start only path yet
           visited: empty list to keep track of visited vertex

    @return path: a list of tuples containing the coordinates of each state in the computed path

    # empty list of path that we should return at the end of this recursive call 
    bfs_path = []

    # if q is not empty we keep searching using BFS
    while q:

        # pop the head of the queue list and put it in the tuple set of a vertex and a path
        (current_position, current_path) = q.pop(0)

        # here we explore our neighbor positions of numbers at maximum of 4 or a minimum of 1, 
        # but exclude taht is already visited or included in the current path. 
        # neighbor: just like i for usual for loop 
        for neighbor in set(maze.getNeighbors(current_position[0], current_position[1])) - set(visited) - set(current_path):
            
            # Case I: when we find an objective we have to find/visit in our list of neighbors(must visit scenario)
            if neighbor in objectives:

                # because we have set the path in the list form, we could simply add our neighbor using + 
                bfs_path = current_path + [neighbor]

                # this is our goal state so if we find one of them here, we take them out from the objectives list
                objectives.remove(neighbor)

                # Case I ended
                if not objectives:
                    return bfs_path

                # enqueue the current position and the path / and mark it as visited by putting into the visited list       
                q.append( (neighbor, current_path + [neighbor]) )
                visited.append(neighbor)

                # recurse on the same fucntion call
                recursive_path = bfs_solve(maze, neighbor, objectives, q, visited)
                return bfs_path + recursive_path
            
            # ELSE CASE
            else:
                q.append((neighbor, current_path + [neighbor]))   
                visited.append(neighbor) 
    
    # return the final bfs_path
    print(bfs_path)
    return bfs_path
"""



# Class for Graph representation 
# MST for A* search algorithm, Kruskal for building MST,  premade from the referrence site
# referrence webpage: https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/
class Graph: 
  
    def __init__(self,vertices): 
        self.V= vertices #No. of vertices 
        self.graph = [] # default dictionary  
                                # to store graph 
          
   
    # function to add an edge to graph 
    def addEdge(self,u,v,w): 
        self.graph.append([u,v,w]) 
  
    # A utility function to find set of an element i 
    # (uses path compression technique) 
    def find(self, parent, i): 
        if parent[i] == i: 
            return i 
        return self.find(parent, parent[i]) 
  
    # A function that does union of two sets of x and y 
    # (uses union by rank) 
    def union(self, parent, rank, x, y): 
        xroot = self.find(parent, x) 
        yroot = self.find(parent, y) 
  
        # Attach smaller rank tree under root of  
        # high rank tree (Union by Rank) 
        if rank[xroot] < rank[yroot]: 
            parent[xroot] = yroot 
        elif rank[xroot] > rank[yroot]: 
            parent[yroot] = xroot 
  
        # If ranks are same, then make one as root  
        # and increment its rank by one 
        else : 
            parent[yroot] = xroot 
            rank[xroot] += 1
  
    # The main function to construct MST using Kruskal's  
        # algorithm 
    def KruskalMST(self): 
  
        result = [] #This will store the resultant MST 
  
        i = 0 # An index variable, used for sorted edges 
        e = 0 # An index variable, used for result[] 
  
            # Step 1:  Sort all the edges in non-decreasing  
                # order of their 
                # weight.  If we are not allowed to change the  
                # given graph, we can create a copy of graph 
        self.graph =  sorted(self.graph,key=lambda item: item[2]) 
  
        parent = [] ; rank = [] 
  
        # Create V subsets with single elements 
        for node in range(self.V): 
            parent.append(node) 
            rank.append(0) 
      
        # Number of edges to be taken is equal to V-1 
        while e < self.V -1 : 
  
            # Step 2: Pick the smallest edge and increment  
                    # the index for next iteration 
            u,v,w =  self.graph[i] 
            i = i + 1
            x = self.find(parent, u) 
            y = self.find(parent ,v) 
  
            # If including this edge does't cause cycle,  
                        # include it in result and increment the index 
                        # of result for next edge 
            if x != y: 
                e = e + 1     
                result.append([u,v,w]) 
                self.union(parent, rank, x, y)             
            # Else discard the edge 
  
        return result 


#helper function to backtrace 
def backtrace(parent, curr):
    path = [curr]
    while curr in parent.keys():
        curr = parent[curr]
        path = [curr] + path
    return path


# chebyshev distance
# referred from theory.stanford.edu/~amitp/GameProgramming/Heuristics.html
def diag_heuristic(vertex, goal):
    dy = abs(vertex[0]-goal[0])
    dx = abs(vertex[1]-goal[1])
    return (dx+dy) + (-1)* min(dx,dy)


def get_path_from_mst(mst, objectives):
    u = {}
    v = {}
    w = {}

    path = { 0: [] }
    path_to_leaf = {    }
    visited = []

    for i in range(0,len(objectives)+1):
        u[i] = []
        v[i] = []
        w[i] = 0
        for edge in mst:
            if edge[0] == i:
                v[i].append((edge[1],edge[2]))
            elif edge[1] == i:
                u[i].append((edge[0],edge[2]))
    mst_path(mst, u, v, w, path, path_to_leaf, visited,0)
    leaf_weight = {}

   # path_lengths = {}
    for leaf in path_to_leaf:
        leaf_weight[leaf] = w[leaf]

    return path_to_leaf

def find_neighbors(u,v,visited):
    neighbors = u+v
    for neighbor in neighbors:
        if neighbor[0] in visited:
            neighbors.remove(neighbor)
    return neighbors

def mst_path(mst,u,v,w,path,path_to_leaf,visited,i):
    neighbors = find_neighbors(u[i],v[i],visited)
    visited.append(i)
    if not neighbors:
        path_to_leaf[i] = path[i]
    for neighbor in neighbors:
        w[neighbor[0]] = w[i]+neighbor[1]
        path[neighbor[0]] = path[i]+[neighbor[0]]
        mst_path(mst,u,v,w,path,path_to_leaf,visited,neighbor[0])


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    start = maze.getStart()
    objectives = maze.getObjectives()
    f_x = {}
    g_x = {}
    g_x[start] = 0
    graph_path = Graph(len(objectives) + 1)


    for i in range(0,len(objectives)):
        f_x = {}
        g_x = {}
        g_x[start] = 0
        graph_path.addEdge(0,i+1,len(astar_recursive(maze,start,[objectives[i]],f_x,g_x)))

        for j in range(1,len(objectives)):
            f_x = {}
            g_x = {}
            g_x[objectives[i]] = 0
            graph_path.addEdge(i+1,j+1,len(astar_recursive(maze,objectives[i],[objectives[j]],f_x,g_x)))


    kruskal_mst = graph_path.KruskalMST()
    
    # print all edges in the mst
    for edge in kruskal_mst:
        if edge[0] == 0:
            v1 = start
        else:
            v1 = objectives[edge[0]-1]
        if edge[1] == 0:
            v2 = start
        else:
            v2 = objectives[edge[1]-1]
    #tiny search mst shortest path.
    path_to_leaf = get_path_from_mst(kruskal_mst,objectives)

    shortest_path = []
    objs = []
    first_run = True
    for leaves in itertools.permutations(path_to_leaf.keys()):
        objectives = maze.getObjectives()
        
        objs = []
        for leaf in leaves:
            for vertex in path_to_leaf[leaf]:
                if objectives[vertex-1] not in objs:
                    objs.append(objectives[vertex-1])
         # objectives = [objectives[2],objectives[7],objectives[5],objectives[1],objectives[6],objectives[10],objectives[3],objectives[8],objectives[9],objectives[11],objectives[4],objectives[0]]
        f_x = {}
        g_x = {}
        g_x[start] = 0
        # start to all objectives
        path = astar_recursive(maze, start, objs, f_x, g_x)

        if len(path) < len(shortest_path) or first_run:
            first_run = False
            shortest_path = path
    return shortest_path
   

def astar_recursive(maze, start, objectives, f_x, g_x):
    open_set = [start]
    closed_set = []
    parent = {}
   
    f_x[start] = diag_heuristic(start, objectives[0])

    while open_set:
    	# the node in the open set having the lowest f cost value
        curr = min(open_set, key=(lambda k: f_x[k]))

        if curr in objectives:
        	# backtrace and find the shortest path
            path = backtrace(parent, curr)
            objectives.remove(curr)
            if not objectives:
                return path
            open_set.remove(curr)
            closed_set.append(curr)
            rec_path = astar_recursive(maze, curr, objectives, f_x, g_x)
            return path + rec_path[1:]

        open_set.remove(curr)
        closed_set.append(curr)

        for neighbor in set(maze.getNeighbors(curr[0], curr[1])):
            # ignore neighbor that has already been visited
            if neighbor in closed_set:
                continue

            # distance from start to neighbor
            temp_g_x = g_x[curr] + 1

            if neighbor not in open_set:
                open_set.append(neighbor)
            elif temp_g_x >= g_x[neighbor]:
                continue

            parent[neighbor] = curr
            g_x[neighbor] = temp_g_x
            f_x[neighbor] = g_x[neighbor] + diag_heuristic(neighbor, objectives[0])

    return path  





def fast(maze):

    start = maze.getStart()
    objectives = maze.getObjectives()
    g_x = {}
    g_x[start] = 0
    f_x = {}
    return fast_recursive(maze, start, objectives, f_x, g_x)

def fast_recursive(maze, start, objectives, f_x, g_x):
    open_set = [start]
    closed_set = []
    parent = {}
    obj_hq = []

    # calculate distance from start to each objective
	# and store them in priority queue by the heuristic value
    for o in objectives:
        heappush(obj_hq, (diag_heuristic(start, o), o))
    
    # objective with the lowest h (closest from start) is popped first
    obj = heappop(obj_hq)
    f_x[start] = diag_heuristic(start, objectives[0])

    while open_set:
    	# the node in the open set having the lowest f cost value
        curr = min(open_set, key=(lambda k: f_x[k]))

        if curr in objectives:
        	# backtrace and find the shortest path
            path = backtrace(parent, curr)
            objectives.remove(curr)
            if not objectives:
                return path
            open_set.remove(curr)
            closed_set.append(curr)
            rec_path = astar_recursive(maze, curr, objectives, f_x, g_x)
            return path + rec_path[1:]

        open_set.remove(curr)
        closed_set.append(curr)

        for neighbor in set(maze.getNeighbors(curr[0], curr[1])):
            # ignore neighbor that has already been visited
            if neighbor in closed_set:
                continue

            # distance from start to neighbor
            temp_g_x = g_x[curr] + 1

            if neighbor not in open_set:
                open_set.append(neighbor)
            elif temp_g_x >= g_x[neighbor]:
                continue

            parent[neighbor] = curr
            g_x[neighbor] = temp_g_x
            f_x[neighbor] = g_x[neighbor] + diag_heuristic(neighbor, objectives[0])

    return path  










def astar_multi(maze):
    return []


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    return []


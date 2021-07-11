# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def bfs(maze):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 
    """
    queue = []
    visited = set()
    start = maze.getStart()
    queue.append([start])

    # this is just simply same as while queue isn't empty
    while queue:
        current_path = queue.pop(0)
        current_row, current_col = current_path[-1]

        #if we visited, continue
        if (current_row, current_col) in visited:
            continue
        visited.add((current_row, current_col))

        #if objective, return the path up until now
        if maze.isObjective(current_row, current_col):
            #print("obj, current path: ", current_path)
            if(maze.isValidPath(current_path) == 'Valid'):
                #print(maze.isValidPath(current_path))
                return current_path
            else: 
                return None
        for neighbor in maze.getNeighbors(current_row, current_col):
            if neighbor not in visited:
                queue.append(current_path + [neighbor])
    
    
    if current_path is []:
        return None

    # while loop broke; we should return None and just in case empty the current_path   
    else:
        #print("out of while loop, current path: ", current_path)
        current_path = []
        return None


# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    # Lists initialization
    dimension = []
    offsets = []
    arm_lengths = []
    
    # brings all initial data that are necessary
    # x,y coordinate of arm base(a.k.a. 1st armlink at the bottom)
    arm_base = arm.getBase()
    # tuples of all x,y coordinates of armlinks
    arm_pos = arm.getArmPos()
    # tuples of all min and max angles
    arm_limits = arm.getArmLimit()
    # alpha and beta
    arm_angles = arm.getArmAngle() 

    # fill up all arm_lengths, remember using distance from geometry.py
    for arm_link in arm_pos:
        arm_len = int(distance(arm_link[0],arm_link[1]))
        arm_lengths.append(arm_len)

    for limit in arm_limits:
        offsets.append(limit[0])
        dim = int(((limit[1]-limit[0])/granularity)+1)
        dimension.append(dim)
        
    # input_map initialization using loop and range in python library
    # very useful to initatilizing the map at the beginning
    # 1-D Case
    if len(dimension) == 1:
        input_map = [SPACE_CHAR]*dimension[0]

    # 2-D Case    
    elif len(dimension) == 2:
        input_map = [[SPACE_CHAR]*dimension[1] for y in range(dimension[0])]

    # 3-D Case     
    else:
        input_map = [[[SPACE_CHAR]*dimension[2] for x in range(dimension[1])] for y in range(dimension[0])]

    # Dimensional work / drawing walls and objectives
    # 1-D Case
    if len(dimension) == 1:

        # we specify starting point to the end point of this for loop 
        for i in range(0, len(input_map)):
            
            # use util.py's idxToAngle getting an alpha angle 
            alpha = idxToAngle([i], offsets, granularity)
            alpha_end = computeCoordinate(arm_base, arm_lengths[0], alpha)
            arm_pos[0] = (arm_base, alpha_end)

            if doesArmTouchObjects(arm_pos, obstacles) or not isArmWithinWindow(arm_pos, window):
                input_map[i] = WALL_CHAR
                continue

            if not doesArmTouchObjects(arm_pos[0][1], goals) and doesArmTouchObjects(arm_pos, goals):
                input_map[i] = WALL_CHAR
                continue
            
            # goals
            if doesArmTipTouchGoals(arm_pos[0][1], goals):
                input_map[i] = OBJECTIVE_CHAR

    # 2-D Case
    elif len(dimension) == 2:
        for i in range(0, len(input_map)):
            for j in range(0, len(input_map[i])):
                (alpha, beta) = idxToAngle([i,j], offsets, granularity)
                alpha_end = computeCoordinate(arm_base, arm_lengths[0], alpha)
                arm_pos[0] = (arm_base, alpha_end)
                beta_end = computeCoordinate(alpha_end, arm_lengths[1], beta+alpha)
                arm_pos[1] = (alpha_end, beta_end)
                
                if doesArmTouchObjects(arm_pos, obstacles) or not isArmWithinWindow(arm_pos, window):
                    input_map[i][j] = WALL_CHAR
                    continue

                if not doesArmTipTouchGoals(arm_pos[1][1], goals) and doesArmTouchObjects(arm_pos, goals):
                    input_map[i][j] = WALL_CHAR
                    continue
               
                if doesArmTipTouchGoals(arm_pos[1][1], goals):
                    input_map[i][j] = OBJECTIVE_CHAR

    # 3-D Case
    else:
        for i in range(0, len(input_map)):
            for j in range(0, len(input_map[i])):
                for k in range(0, len(input_map[i][j])):
                    (alpha, beta, gamma) = idxToAngle([i,j,k], offsets, granularity)
                    alpha_end = computeCoordinate(arm_base,arm_lengths[0], alpha)
                    arm_pos[0] = (arm_base, alpha_end)
                    beta_end = computeCoordinate(alpha_end,arm_lengths[1], beta+alpha)
                    arm_pos[1] = (alpha_end, beta_end)
                    gamma_end = computeCoordinate(beta_end,arm_lengths[2], beta+alpha+gamma)
                    arm_pos[2] = (beta_end, gamma_end)

                    if doesArmTouchObjects(arm_pos, obstacles) or not isArmWithinWindow(arm_pos, window):
                        input_map[i][j][k] = WALL_CHAR
                        continue

                    if not doesArmTipTouchGoals(arm_pos[2][1], goals) and doesArmTouchObjects(arm_pos, goals):
                        input_map[i][j][k] = WALL_CHAR
                        continue
                    
                    if doesArmTipTouchGoals(arm_pos[2][1], goals):
                        input_map[i][j][k] = OBJECTIVE_CHAR

    # Starting point setup
    starting = angleToIdx(arm_angles, offsets, granularity)
    #1-D Case
    if len(dimension) == 1:
        input_map[starting[0]] = START_CHAR

    #2-D Case    
    elif len(dimension) == 2:
        input_map[starting[0]][starting[1]] = START_CHAR

    #3-D Case     
    else:
        input_map[starting[0]][starting[1]][starting[2]] = START_CHAR

    return Maze(input_map, offsets, granularity)

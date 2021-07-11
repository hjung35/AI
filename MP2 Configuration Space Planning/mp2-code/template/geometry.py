# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *

def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise

        Return:
            End position (int,int):of the arm link, (x-coordinate, y-coordinate)
    """

    rad = math.radians(angle)
    x = length * math.cos(rad)
    y = length * math.sin(rad)
    return (int(start[0]+x), int(start[1]-y))

def doesArmTouchObjects(armPosDist, objects, isGoal=False):
    """Determine whether the given arm links touch any obstacle or goal

        Args:
            armPosDist (list): start and end position and padding distance of all arm links [(start, end, distance)]
            objects (list): x-, y- coordinate and radius of object (obstacles or goals) [(x, y, r)]
            isGoal (bool): True if the object is a goal and False if the object is an obstacle.
                           When the object is an obstacle, consider padding distance.
                           When the object is a goal, no need to consider padding distance.
        Return:
            True if touched. False if not.
    """
     # referrence: http://mathforum.org/library/drmath/view/55188.html
    # https://www.enotes.com/homework-help/ax-by-c-sqrt-2-b-2-could-you-explain-what-437867?__cf_chl_captcha_tk__=abd8f4e44489adfd20bd9d058122acb65b81eb47-1601023471-0-Acp2k8McJosbuSqOv77oQTnRCxfh1QfD8daiEot4oTa8qCh3oUDAi9v8kxeSY8hfVyD6eRSoH8gCGVe5dahTPLRW3RUdVADMjzfFIvtCIwqTmekM8-KEovdeT3nIogDxhwoSs3X8TMj3NhDdO-Vsfb6aYd3wT0XElLd7bmk1ieIXkwDYaUsU38j0fpuwfqm_DQEqkOTCmojDxEB2yLYXCvHIOVdQrxZyU0ms8V2OgaOZ5Knf-Q1bs8ChuEdtzj8qAUPQrAms5N2b7NMnp4Gtgv2ufDRYftXscyVtJWmoZ9LGK-bxHuFSwf9eplqMYhde6ywP30l67RD8I27RDKEOOVIEWIVZU5ylnm4NZZYhwym2XgMyt4ISwNeRULQSlyUygyZv6nDyWVGjz8WFBg_wQ6OgU5prxHOkhQKeEB0K9a4HFTcCh3a_fHwOHhDaNw2WJYtJhA0wExmTWHIEaE4MCnkQP-sKauFhDNZnnnLRz1F5Nc8J3StDIxEM2_ytzHJZ-6qDsJXKBFa1MmG41tF1L_EeZWjHtJY1TNfHDGq7_I3nB1wsHYgAvOOdDuiAhD2uxQ
    for arm in armPosDist:
        #print("arm", arm, "arm[0]: ", arm[0], "arm[1]", arm[1])
        arm_length = distance(arm[0],arm[1])
        a, b, c = line_in_tuple(arm[0], arm[1])
        #padding = 0 if isGoal else arm[2]
        #print(padding)

        # we dont know whether objects is an obstacle or a goal 
        for obj in objects:
            #print(obj)
            # determine whether obj is a goal or not    

            fromStart = distance(arm[0],obj)
            fromEnd = distance(arm[1],obj)
            
            # obj[0]=x, obj[1]=y
            distance_from_center = distance_from_pt_to_line(a, b, c, obj[0], obj[1])

            #obj[2] = radius
            if distance_from_center <= (obj[2]) and fromEnd + fromStart < arm_length + obj[2]:
                return True

    return False

def line_in_tuple(start, end):
    a = (end[1] - start[1])
    b = (start[0] - end[0])
    c = -(a*start[0] + b*start[1])
    return (a, b, c)

def distance_from_pt_to_line(a, b, c, x, y):
    return (abs(a * x + b * y + c)) / math.sqrt(a * a + b * b)


def doesArmTipTouchGoals(armEnd, goals):
    """Determine whether the given arm tip touch goals

        Args:
            armEnd (tuple): the arm tip position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]. There can be more than one goal.
        Return:
            True if arm tip touches any goal. False if not.
    """

    for goal in goals:
        #find if the distance of the goal is lees than or equal to the radius
        if distance(armEnd, goal) <= goal[2]:
            return True
    return False

def distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)    


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end positions of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False if not.
    """

    # Checki the boundary condition and see if it falls outside the boundary
    for arm in armPos:
        if (arm[0][0] < 0) or (arm[0][1] < 0) or (arm[1][0] < 0) or (arm[1][1] < 0) or (arm[0][0] > window[0]) or (arm[1][0] > window[0]) or (arm[0][1] > window[1]) or (arm[1][1] > window[1]):
           return False
    return True


if __name__ == '__main__':
    computeCoordinateParameters = [((150, 190),100,20), ((150, 190),100,40), ((150, 190),100,60), ((150, 190),100,160)]
    resultComputeCoordinate = [(243, 156), (226, 126), (200, 104), (57, 156)]
    testRestuls = [computeCoordinate(start, length, angle) for start, length, angle in computeCoordinateParameters]
    assert testRestuls == resultComputeCoordinate

    testArmPosDists = [((100,100), (135, 110), 4), ((135, 110), (150, 150), 5)]
    testObstacles = [[(120, 100, 5)], [(110, 110, 20)], [(160, 160, 5)], [(130, 105, 10)]]
    resultDoesArmTouchObjects = [
        True, True, False, True, False, True, False, True,
        False, True, False, True, False, False, False, True
    ]

    testResults = []
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle))

    print("\n")
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))

    assert resultDoesArmTouchObjects == testResults

    testArmEnds = [(100, 100), (95, 95), (90, 90)]
    testGoal = [(100, 100, 10)]
    resultDoesArmTouchGoals = [True, True, False]

    testResults = [doesArmTipTouchGoals(testArmEnd, testGoal) for testArmEnd in testArmEnds]
    assert resultDoesArmTouchGoals == testResults

    testArmPoss = [((100,100), (135, 110)), ((135, 110), (150, 150))]
    testWindows = [(160, 130), (130, 170), (200, 200)]
    resultIsArmWithinWindow = [True, False, True, False, False, True]
    testResults = []
    for testArmPos in testArmPoss:
        for testWindow in testWindows:
            testResults.append(isArmWithinWindow([testArmPos], testWindow))
    assert resultIsArmWithinWindow == testResults

    print("Test passed\n")

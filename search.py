# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from game import Directions
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    #Search the deepest nodes in the search tree first.
    stack=util.Stack()
    visited=set()

    stack.push([problem.getStartState(),[]])

    while not stack.isEmpty():
        cur_vertex,vertex_directions=stack.pop()
        if problem.isGoalState(cur_vertex):
            return vertex_directions
        if cur_vertex not in visited:
            visited.update([cur_vertex])
            for child,action,stepCost in problem.expand(cur_vertex):
                if child not in visited:
                    stack.push([child,vertex_directions+[action]])

def breadthFirstSearch(problem):
    #Search the shallowest nodes in the search tree first.
    queue=util.Queue()
    visited=set()

    queue.push([problem.getStartState(),[]])
    visited.update([problem.getStartState()])

    while not queue.isEmpty():
        cur_vertex,vertex_directions=queue.pop()
        if problem.isGoalState(cur_vertex):
            return vertex_directions
        for child,action,stepCost in problem.expand(cur_vertex):
            if child not in visited:
                queue.push([child,vertex_directions+[action]])
                visited.update([child])
    
    

def nullHeuristic(state, problem=None):
    # A heuristic function estimates the cost from the current state to the nearest
    # goal in the provided SearchProblem.  This heuristic is trivial.
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    #Search the node that has the lowest combined cost and heuristic first.
    pq=util.PriorityQueue()
    costs={}

    pq.push([problem.getStartState(),[]],0)
    costs[problem.getStartState()]=0

    while not pq.isEmpty():
        cur_vertex,vertex_directions=pq.pop()
        if problem.isGoalState(cur_vertex):
            return vertex_directions
        for child,action,stepCost in problem.expand(cur_vertex):
            current_cost=costs[cur_vertex]+stepCost
            if child not in costs or current_cost<costs[child]:
                costs[child]=current_cost
                pq.push([child,vertex_directions+[action]],current_cost+heuristic(child,problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch

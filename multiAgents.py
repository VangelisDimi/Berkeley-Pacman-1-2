# multiAgents.py
# --------------
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


from autograder import printTest
from util import manhattanDistance
from game import Actions, Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        oldFood=currentGameState.getFood()

        if len(newFood.asList())==0:
            return float('inf')
        for i in range(len(newGhostStates)):
            if manhattanDistance(newGhostStates[i].getPosition(),newPos) <= newScaredTimes[i]+1:
                return float('-inf')

        score=0
        if len(newFood.asList()) < len(oldFood.asList()):
            score+=1
        min_distance=float('inf')
        for i in range(len(newFood.asList())):
            distance=manhattanDistance(newFood.asList()[i],newPos)
            if distance<min_distance:
                min_distance=distance
        score+=1/min_distance

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    depth=0
    evalutationFunction=scoreEvaluationFunction
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        legalMoves = gameState.getLegalActions(0)
        scores = [self.Min_Value(gameState.getNextState(0,action),self.depth) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]
    
    def Max_Value(self,gameState,depth):
        if depth==0 or gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(0)
        scores = [self.Min_Value(gameState.getNextState(0,action),depth) for action in legalMoves]
        return max(scores)

    def Min_Value(self,gameState,depth,GhostMove=1):
        if depth==0 or gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)

        scores=[]
        legalMoves = gameState.getLegalActions(GhostMove)
        if GhostMove==gameState.getNumAgents()-1:
            scores = [self.Max_Value(gameState.getNextState(GhostMove,action),depth-1) for action in legalMoves]
        else:
            scores = [self.Min_Value(gameState.getNextState(GhostMove,action),depth,GhostMove+1) for action in legalMoves]
        return min(scores)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    depth=0
    evalutationFunction=scoreEvaluationFunction
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        legalMoves = gameState.getLegalActions(0)

        alpha=float('-inf')
        scores=[]
        for action in legalMoves:
            scores += [self.Min_Value(gameState.getNextState(0,action),self.depth,alpha)]
            alpha = max(alpha,max(scores))
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]
    
    def Max_Value(self,gameState,depth,alpha=float('-inf'),beta=float('inf')):
        if depth==0 or gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)

        max_score=float('-inf')
        legalMoves = gameState.getLegalActions(0)
        for action in legalMoves:
            max_score = max(max_score,self.Min_Value(gameState.getNextState(0,action),depth,alpha,beta))
            if max_score > beta:
                return max_score
            alpha=max(alpha,max_score)
        return max_score

    def Min_Value(self,gameState,depth,alpha=float('-inf'),beta=float('inf'),GhostMove=1):
        if depth==0 or gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)

        min_score=float('inf')
        legalMoves = gameState.getLegalActions(GhostMove)
        for action in legalMoves:
            if GhostMove==gameState.getNumAgents()-1:
                min_score = min(min_score,self.Max_Value(gameState.getNextState(GhostMove,action),depth-1,alpha,beta))
            else:
                min_score = min(min_score,self.Min_Value(gameState.getNextState(GhostMove,action),depth,alpha,beta,GhostMove+1))

            if min_score < alpha :
                return min_score
            beta = min(beta,min_score)
        return min_score

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    depth=0
    evalutationFunction=scoreEvaluationFunction
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        legalMoves = gameState.getLegalActions(0)
        scores = [self.Min_Value(gameState.getNextState(0,action),self.depth) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]
        
    def Max_Value(self,gameState,depth):
        if depth==0 or gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(0)
        scores = [self.Min_Value(gameState.getNextState(0,action),depth) for action in legalMoves]
        return max(scores)

    def Min_Value(self,gameState,depth,GhostMove=1):
        if depth==0 or gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)

        scores=[]
        legalMoves = gameState.getLegalActions(GhostMove)
        if GhostMove==gameState.getNumAgents()-1:
            scores = [self.Max_Value(gameState.getNextState(GhostMove,action),depth-1) for action in legalMoves]
        else:
            scores = [self.Min_Value(gameState.getNextState(GhostMove,action),depth,GhostMove+1) for action in legalMoves]
        return sum(scores)/len(scores)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    Capsules = currentGameState.getCapsules()
    PacmanScore = currentGameState.getScore()

    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    score=0
    min_distance_food=float('inf')
    for i in range(len(Food.asList())):
        distance=manhattanDistance(Food.asList()[i],Pos)
        if distance<min_distance_food:
            min_distance_food=distance
    score+=1/min_distance_food

    for i in range(len(Capsules)):
        if manhattanDistance(Capsules[i],Pos)<=2:
            score+=1

    if PacmanScore != 0:
        score+=1/(1/PacmanScore)

    return score

# Abbreviation
better = betterEvaluationFunction

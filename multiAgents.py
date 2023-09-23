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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distToGhost = []
        distToFood = []

        for ghost in newGhostStates:
            distToGhost.append(util.manhattanDistance(newPos, ghost.getPosition()))

        for food in newFood.asList():
            distToFood.append(util.manhattanDistance(newPos, food))

        if len(distToFood) > 0:
            nearestFood = min(distToFood)
            foodH = 10 / nearestFood
        else:
            foodH = 0

        nearestGhost = min(distToGhost)
        if nearestGhost <= 1:
            isDanger = 1
        else:
            isDanger = 0

        return successorGameState.getScore()- 50*isDanger + foodH

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pac5man, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(depth, agent, gameState: GameState):
            # if the state is terminal state, return the state's utility
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            # if the next agent is pacman, find the max_value
            if agent == 0:
                value = float("-inf")
                for nextAction in gameState.getLegalActions(agent):
                    value = max(value, minimax(depth, 1, gameState.generateSuccessor(agent, nextAction)))
                return value
            # if the next agent is ghost, find the min_value
            else:
                value = float("inf")
                nextAgent = agent+1
                if nextAgent%gameState.getNumAgents() == 0:
                    nextAgent = 0
                    depth += 1
                for nextAction in gameState.getLegalActions(agent):
                    value = min(value, minimax(depth, nextAgent, gameState.generateSuccessor(agent, nextAction)))
                return value

        bestAction = None
        bestValue = float("-inf")
        pacmanActions = gameState.getLegalActions(0)
        for action in pacmanActions:
            value = minimax(0, 1, gameState.generateSuccessor(0, action))
            if bestValue < value:
                bestValue = value
                bestAction = action
        return bestAction

          




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(alpha, beta, depth, agent, gameState: GameState):
            # if the state is terminal state, return the state's utility
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            # if the next agent is pacman, find the max_value
            if agent == 0:
                value = float("-inf")
                for nextAction in gameState.getLegalActions(agent):
                    value = max(value, alphabeta(alpha, beta, depth, 1, gameState.generateSuccessor(agent, nextAction)))
                    if value > beta:
                         return value
                    alpha = max(alpha, value)
                return value


            # if the next agent is ghost, find the min_value
            else:
                value = float("inf")
                nextAgent = agent + 1
                if nextAgent % gameState.getNumAgents() == 0:
                    nextAgent = 0
                    depth += 1
                for nextAction in gameState.getLegalActions(agent):
                    value = min(value, alphabeta(alpha, beta, depth, nextAgent, gameState.generateSuccessor(agent, nextAction)))
                    if value < alpha:
                         return value
                    beta = min(beta, value)
                return value


        bestAction = None
        bestValue = float("-inf")
        pacmanActions = gameState.getLegalActions(0)
        for action in pacmanActions:
            value = alphabeta(float("-inf"), float("inf"), 0, 1, gameState.generateSuccessor(0, action))
            if bestValue < value:
                bestValue = value
                bestAction = action
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

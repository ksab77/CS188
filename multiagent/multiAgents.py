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
import math

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        newCapsules = successorGameState.getCapsules()

        "*** YOUR CODE HERE ***"
        pacGhostDist = float(sum([math.sqrt(manhattanDistance(newPos, ghostPos)) for ghostPos in successorGameState.getGhostPositions()]))
        pacFoodDist = float(sum([math.sqrt(manhattanDistance(newPos, foodPos)) for foodPos in newFood.asList()]))
        if (newCapsules):
            pacCapsuleDist = sum([manhattanDistance(newPos, capsulePos) for capsulePos in newCapsules])
        else:
            pacCapsuleDist = 0
            
        
        score = (1.0 / (len(newFood.asList())+1)) + (100.0 / (pacFoodDist+1)) - (3.5 / (pacGhostDist+1))
        #if (sum(newScaredTimes) > 0):
            #score += 0.1 / pacCapsuleDist
            
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
    
    def MiniMaxHelper(self, state, agent, ply):
        if (state.isWin() | state.isLose()):
            return self.evaluationFunction(state)
     
        agentMoves = state.getLegalActions(agent)
        agentStates = [state.generateSuccessor(agent, move) for move in agentMoves]
        agentNum = state.getNumAgents()

        if ((agent == agentNum - 1) & (ply == self.depth)):
            terminalState = util.PriorityQueue()
            for agentState in agentStates:
                terminalState.push(agentState, self.evaluationFunction(agentState))
            return self.evaluationFunction(terminalState.pop())
     
        elif (agent == 0):
            pacMax = util.PriorityQueue()
            for agentState in agentStates:
                minimax = self.MiniMaxHelper(agentState, (agent+1)%agentNum, ply)
                pacMax.push(minimax, -minimax)
            return pacMax.pop()

        elif ((agent != 0)):
            lastGhost = (agent == (agentNum - 1))
            ghostMin = util.PriorityQueue()
            for agentState in agentStates:
                minimax = self.MiniMaxHelper(agentState, (agent+1)%agentNum, ply+lastGhost)
                ghostMin.push(minimax, minimax)
            return ghostMin.pop()

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

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
        
        pacMoves = gameState.getLegalActions(0)
        pacStates = [gameState.generateSuccessor(0, move) for move in pacMoves]
        for pacState in pacStates:
            pacStates[pacStates.index(pacState)] = self.MiniMaxHelper(pacState, 1, 1)
 
        return pacMoves[pacStates.index(max(pacStates))]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def AlphaBetaHelper(self, state, agent, ply, agentMin, agentMax):
        
        if (state.isWin() | state.isLose()):
            return self.evaluationFunction(state)
     
        agentMoves = state.getLegalActions(agent)
        agentNum = state.getNumAgents()

        if ((agent == agentNum - 1) & (ply == self.depth)):
            terminalState = util.PriorityQueue()
            for move in agentMoves:
                agentState = state.generateSuccessor(agent, move)
                terminalState.push(agentState, self.evaluationFunction(agentState))
                if ((agentMax != float("inf")) & (self.evaluationFunction(agentState) < agentMax)):
                    break
                
            terminalState = terminalState.pop()
            return self.evaluationFunction(terminalState)
     
        elif (agent == 0):
            pacMax = util.PriorityQueue()
            currMin, currMax = agentMin, agentMax
            for move in agentMoves:
                agentState = state.generateSuccessor(agent, move)
                terminal = not (agentState.isWin() | agentState.isLose())
                minimax = self.AlphaBetaHelper(agentState, (agent+terminal)%agentNum, ply, agentMin, agentMax)
                pacMax.push(minimax, -minimax)
                if ((currMin != -float("inf")) & (minimax > currMin)):
                    break
                agentMax = max(agentMax, minimax)
            pacMax = pacMax.pop()
            return pacMax

        elif (agent != 0):
            lastGhost = (agent == (agentNum - 1))
            ghostMin = util.PriorityQueue()
            currMin, currMax = agentMin, agentMax
            for move in agentMoves:
                agentState = state.generateSuccessor(agent, move)
                terminal = not (agentState.isWin() | agentState.isLose())
                minimax = self.AlphaBetaHelper(agentState, (agent+terminal)%agentNum, ply+(lastGhost&terminal), agentMin, agentMax)
                ghostMin.push(minimax, minimax)
                if ((currMax != float("inf")) & (minimax < currMax)):
                    break
                agentMin = min(agentMin, minimax)
            ghostMin = ghostMin.pop()
            return ghostMin

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacMoves = gameState.getLegalActions(0)
        pacStates = [gameState.generateSuccessor(0, move) for move in pacMoves]
        agentMin = float("inf")
        agentMax = -float("inf")
        for pacState in pacStates:
            score = self.AlphaBetaHelper(pacState, 1, 1, agentMin, agentMax)
            pacStates[pacStates.index(pacState)] = score
            agentMax = max(agentMax, score)
        return pacMoves[pacStates.index(max(pacStates))]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def ExpectimaxHelper(self, state, agent, ply):
        if (state.isWin() | state.isLose()):
            return self.evaluationFunction(state)
     
        agentMoves = state.getLegalActions(agent)
        agentStates = [state.generateSuccessor(agent, move) for move in agentMoves]
        agentNum = state.getNumAgents()

        if ((agent == agentNum - 1) & (ply == self.depth)):
            terminalState = list()
            for agentState in agentStates:
                terminalState.append(self.evaluationFunction(agentState))
            return sum(terminalState) / len(terminalState)
     
        elif (agent == 0):
            pacMax = util.PriorityQueue()
            for agentState in agentStates:
                minimax = self.ExpectimaxHelper(agentState, (agent+1)%agentNum, ply)
                pacMax.push(minimax, -minimax)
            return pacMax.pop()

        elif ((agent != 0)):
            lastGhost = (agent == (agentNum - 1))
            ghostMin = list()
            for agentState in agentStates:
                minimax = self.ExpectimaxHelper(agentState, (agent+1)%agentNum, ply+lastGhost)
                ghostMin.append(minimax)
            return sum(ghostMin) / len(ghostMin)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        pacMoves = gameState.getLegalActions(0)
        pacStates = [gameState.generateSuccessor(0, move) for move in pacMoves]
        for pacState in pacStates:
            pacStates[pacStates.index(pacState)] = self.ExpectimaxHelper(pacState, 1, 1)
 
        return pacMoves[pacStates.index(max(pacStates))]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsules = currentGameState.getCapsules()

    
    pacGhostDist = float(sum([math.sqrt(manhattanDistance(newPos, ghostPos)) for ghostPos in currentGameState.getGhostPositions()]))
    pacFoodDist = float(sum([math.sqrt(manhattanDistance(newPos, foodPos)) for foodPos in newFood.asList()]))
    if (newCapsules):
        pacCapsuleDist = sum([manhattanDistance(newPos, capsulePos) for capsulePos in newCapsules])
    else:
        pacCapsuleDist = 0
            
        
    score = (1.0 / (len(newFood.asList())+1)) + (100.0 / (pacFoodDist+1))
    if (sum(newScaredTimes) > 2):
        score += (6.0 / (pacGhostDist+1))
    else:
        score += 3.0 / (pacCapsuleDist+1) - (6.0 / (pacGhostDist+1))
            
    return score

# Abbreviation
better = betterEvaluationFunction


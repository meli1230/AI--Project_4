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
        # @Author: Marian Melisa

        all_food = newFood.asList() #get all food positions from the successorGameState

        if all_food:
            distance_to_all_food = [manhattanDistance(newPos, food_item) for food_item in all_food] #compute the distances to each food item
            closest_food_item = min(distance_to_all_food) #get the closest food item
            successorGameState.data.score += 10/(1+closest_food_item) #the closer to food pacman is, the higher the reward
            successorGameState.data.score -= len(all_food)*5 #penalize pacman for each food that has not been eaten yet
                                                        #this line is not absolutely necessary; if we omit it, we get the 4th score under 1000, which does not affect the max grade
                                                        #however, keeping it in gets us a better performance of pacman

        for i in range(len(newGhostStates)):
            position_of_ghost = newGhostStates[i].getPosition() #get ghost state
            distance_to_ghost = manhattanDistance(newPos, position_of_ghost) #calculate the distance to ghost
            if newScaredTimes[i] > 0: #check if the ghost is in the scared state
                successorGameState.data.score += 100/(1+distance_to_ghost)
            elif distance_to_ghost < 2: #if the ghost is not scared and the ghost is close
                    successorGameState.data.score -= 1000 #penalize to avoid it

        return successorGameState.getScore()

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
        # @Author: Melisa Marian


        def mini(ghost_position, current_depth, gameState):
            #minimize the score for ghosts
            possible_moves = gameState.getLegalActions(ghost_position)
            if not possible_moves:
                return self.evaluationFunction(gameState)

            ghost_score = float('inf') #initialize ghost_score to infinity, which is the best case for pacman
            number_of_agents = gameState.getNumAgents() #get the number of agents (pacman + ghosts)
            next_agent_position = (ghost_position+1) % number_of_agents #get the agent that is about to make a move

            if next_agent_position == 0:  #if agent is pacman
                next_depth = current_depth + 1 #increment depth
            else: #if agent is not pacman
                next_depth = current_depth #depth stays the same

            for move in possible_moves:
                next_gameState = gameState.generateSuccessor(ghost_position, move) #generate successor state
                agent_score = minimax (next_agent_position, next_depth, next_gameState) #recursive call to minimax
                ghost_score = min(ghost_score, agent_score) #update the lowest score

            return ghost_score

        def maxi(pacman_position, current_depth, gameState):
            #maximize the score for Pacman
            possible_moves = gameState.getLegalActions(pacman_position)
            if not possible_moves:
                return self.evaluationFunction(gameState)

            pacman_score = float('-inf')  # initialize pacman_score to negative infinity, which is the worst case for pacman

            for move in possible_moves:
                next_gameState = gameState.generateSuccessor(pacman_position, move)  #generate successor state
                agent_score = minimax(1, current_depth, next_gameState)  #recursive call to minimax
                pacman_score = max(pacman_score, agent_score)  #update the highest score

            return pacman_score


        def minimax (agent_position, current_depth, gameState):
            if gameState.isWin() or gameState.isLose() or current_depth == self.depth: #checks if the game has reached the finish line
                return self.evaluationFunction(gameState)

            if agent_position == 0: #checks if it is pacman's turn
                return maxi (agent_position, current_depth, gameState)
            else:
                return mini (agent_position, current_depth, gameState)


        #logic of getAction()
        possible_moves = gameState.getLegalActions(0) #pacman = agent 0

        best_move_to_make = None #initialize best move to null
        best_score = float('-inf') #initialize best score to negative infinity

        for move in possible_moves:
            next_state = gameState.generateSuccessor(0, move) #generate game state after making a move
            move_rating = minimax(1, 0, next_state) #evaluate this move with minimax
                                                                        # 1 -> ghost is 1, since pacman is 0
                                                                        # 0 -> current depth of minimax

            if move_rating > best_score: #if the score for this is better than the best score
                best_score = move_rating #update best score
                best_move_to_make = move #update best move

        return best_move_to_make


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # @Author: Melisa Marian

        def mini(ghost_position, score, state, alpha, beta):
            #maximize the score for ghosts
            ghost_score = float('inf')  #initialize ghost_score to infinity, which is the best case for pacman
            possible_moves = state.getLegalActions(ghost_position)

            if not possible_moves:
                return self.evaluationFunction(state)

            next_agent_position = ghost_position + 1 #get the agent that is about to make a move

            if next_agent_position == state.getNumAgents():  #check if last move was made my ghost
                next_agent_position = 0  #agent is now pacman
                score += 1  #increment depth

            for move in possible_moves:
                next_gameState = state.generateSuccessor(ghost_position, move) #generate successor state
                ghost_score = min(ghost_score, alphaBetaPruning(next_agent_position, score, next_gameState, alpha, beta)) #recursive call to abpruning

                if ghost_score < alpha: #check if the lowest score is less than the alpha threshold
                    return ghost_score  #alpha pruning

                beta = min(beta, ghost_score) #update the beta threshold to the minimum of its current value

            return ghost_score


        def maxi(pacman_position, score, state, alpha, beta):
            #maximize the score for pacman
            pacman_score = float('-inf') #initialize pacman_score to negative infinity, which is the worst case for pacman
            possible_moves = state.getLegalActions(pacman_position)

            if not possible_moves:
                return self.evaluationFunction(state)

            for move in possible_moves:
                next_GameState = state.generateSuccessor(pacman_position, move)
                pacman_score = max(pacman_score, alphaBetaPruning(1, score, next_GameState, alpha, beta)) #recursive call to abpruning

                if pacman_score > beta: #check if the highest score is more than the beta threshold
                    return pacman_score  #beta pruning

                alpha = max(alpha, pacman_score) #update the alpha threshold to the maximum of it current value

            return pacman_score



        def alphaBetaPruning(agent_position, score, state, alpha, beta):
            if state.isWin() or state.isLose() or score == self.depth: #if the game is finished
                return self.evaluationFunction(state)  #the maximum depth is reached

            if agent_position == 0: #check if it is pacman's turn
                return maxi(agent_position, score, state, alpha, beta)
            else:
                return mini(agent_position, score, state, alpha, beta)


        #logic of getAction()
        alpha_maxi = float('-inf') #initialize alpha to negative infinity
        beta_mini = float('inf') #initialize beta to infinity
        best_move_to_make = None #initialize best move to null
        possible_moves_to_make = gameState.getLegalActions(0) #get the possible moves pacman can make

        for move in possible_moves_to_make:
            next_state = gameState.generateSuccessor(0, move) #generate game state after making a move
            move_rating = alphaBetaPruning(1, 0, next_state, alpha_maxi, beta_mini) #evaluate this move with abpruning

            if move_rating > alpha_maxi:  #if the score for this is better than the best score
                alpha_maxi = move_rating  #update alpha
                best_move_to_make = move  #update best move

        return best_move_to_make



# Iulia Anca
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
        # util.raiseNotDefined()

        actiuniLegale = gameState.getLegalActions(0)
        if not actiuniLegale:
            return Directions.STOP

        scoruri = []
        for actiune in actiuniLegale:
            succesor = gameState.generateSuccessor(0, actiune)
            scor = self.calculeazaExpectimax(succesor, self.depth, 1)
            scoruri.append(scor)
            
        # scorMaxim = max(scoruri)

        bestActions = []
        for index, scor in enumerate(scoruri):
            if scor == max(scoruri):
                bestActions.append(index)
                
        choice = random.choice(bestActions)

        return actiuniLegale[choice]

    def calculeazaExpectimax(self, stare, adancime, indexAgent):
        if stare.isWin() or stare.isLose() or adancime == 0:
            return self.evaluationFunction(stare)

        if indexAgent == 0:
            valoareMaxima = float('-inf')
            actiuni = stare.getLegalActions(indexAgent)
            for actiune in actiuni:
                succesor = stare.generateSuccessor(indexAgent, actiune)
                scor = self.calculeazaExpectimax(succesor, adancime, 1)
                if scor > valoareMaxima:
                    valoareMaxima = scor
            return valoareMaxima

        nextAgent = indexAgent + 1
        if nextAgent == stare.getNumAgents():
            nextAgent = 0
            adancime -= 1

        actiuni = stare.getLegalActions(indexAgent)
        if not actiuni:
            return self.evaluationFunction(stare)

        probabilitate = 1 / len(actiuni)

        sumaValori = 0
        actiuni = stare.getLegalActions(indexAgent)
        for actiune in actiuni:
            succesor = stare.generateSuccessor(indexAgent, actiune)
            scor = self.calculeazaExpectimax(succesor, adancime, nextAgent)
            sumaValori += scor * probabilitate

        return sumaValori
        




# Iulia Anca
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    capsule = currentGameState.getCapsules()
    scorCurent = currentGameState.getScore()
    pozitiePacman = currentGameState.getPacmanPosition()
    stariFantome = currentGameState.getGhostStates()
    mancare = currentGameState.getFood().asList()

    scorCapsule = 0
    if capsule:

        distanteCapsule = []
        for capsula in capsule:
            distanta = manhattanDistance(capsula, pozitiePacman)
            distanteCapsule.append(distanta)

        distantaMinimaCapsula = min(distanteCapsule)
        scorCapsule = 25.0 / (1 + distantaMinimaCapsula)

    scorMancare = 0
    if mancare:

        distanteMancare = []
        for manc in mancare:
            distanteMancare.append(manhattanDistance(manc, pozitiePacman))

        distantaMinimaMancare = min(distanteMancare)
        scorMancare = 5.0 / (1 + distantaMinimaMancare)

    penalizareFantome = 0
    for fantoma in stariFantome:
        pozitieFantoma = fantoma.getPosition()
        distantaFantoma = manhattanDistance(pozitiePacman, pozitieFantoma)
        if fantoma.scaredTimer > 0:
            penalizareFantome += 300.0 / (1 + distantaFantoma)
        elif distantaFantoma < 2:
            penalizareFantome -= 1000.0 / (1 + distantaFantoma)

    return scorCurent + scorCapsule + scorMancare + penalizareFantome

# Abbreviation
better = betterEvaluationFunction

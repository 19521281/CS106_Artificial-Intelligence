import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = [] #initiate an empty heap
        self.Count = 0 #initiate a variable "Count" with 0
        self.len = 0 #initiate the length of heap with 0

    def push(self, item, priority):
        entry = (priority, self.Count, item) #assign 3 values to be inserted in variable "entry'
        heapq.heappush(self.Heap, entry) #push 3 values of variable "entry' into heap
        self.Count += 1 #increase variable "Count" by 1 unit

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)  #choose an item which is equal to the smallest cost popped from heap
        return item #return item which has the smallest cost.

    
    def isEmpty(self): 
        return len(self.Heap) == 0 #return the number of len is 0 if the queue has no elements after checking

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)

def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     
    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState) #beginBox contains all position of boxes
    beginPlayer = PosOfPlayer(gameState) #beginPlayer contains position of player

    startState = (beginPlayer, beginBox) #startState contains the initial position of boxes and player
    frontier = collections.deque([[startState]]) #store states
    exploredSet = set() #initiate a variable "exploredSet" to map all environment and check steps exited or not then skip or add
    actions = [[0]] #initiate a variable "actions" with 0 
    temp = [] #initiate an empty temp
    while frontier: 
        node = frontier.pop() #node is the first element popped on the right of queue
        node_action = actions.pop() #node_action is the first action popped on the right of queue
        if isEndState(node[-1][-1]): #check if the solution found out (boxes are at goal positions)
            temp += node_action[1:] #return a list of actions that completed games
            break
        if node[-1] not in exploredSet: #if found node still not checked yet
            exploredSet.add(node[-1]) #add to a variable "exploredSet"
            for action in legalActions(node[-1][0], node[-1][1]): #initiate an iteration with all posible actions of current posPlayer and posBox
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) #update new state for posPlayer and posBox
                if isFailed(newPosBox): #if new state is invalid skip
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)]) #append new position of player and box into frontier
                actions.append(node_action + [action[-1]]) #append new action had been done into actions
    return temp #list actions to get success

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState) #beginBox contains all position of boxes
    beginPlayer = PosOfPlayer(gameState) #beginPlayer contains position of player

    startState = (beginPlayer, beginBox) # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    frontier = collections.deque([[startState]]) # store states
    actions = collections.deque([[0]]) # store actions
    exploredSet = set()  #initiate a variable "exploredSet" to map all environment and check steps exited or not then skip or add
    temp = [] #initiate an empty temp
    ### Implement breadthFirstSearch here
    while frontier:
        node = frontier.popleft() #node is the first element popped on the left of queue
        node_action = actions.popleft() #node_action is the first action popped on the left of queue
        if isEndState(node[-1][-1]): #check if the solution found out (boxes are at goal positions)
            temp += node_action[1:] #return a list of actions that completed games
            break
        if node[-1] not in exploredSet: #if found node still not checked yet
            exploredSet.add(node[-1]) #add to a variable "exploredSet"
            for action in legalActions(node[-1][0], node[-1][1]): #initiate an iteration with all posible actions of current posPlayer and posBox
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) #update new state for posPlayer and posBox
                if isFailed(newPosBox): #if new state is invalid, skip
                    continue 
                frontier.append(node + [(newPosPlayer, newPosBox)]) #append new position of box and player into frontier
                actions.append(node_action + [action[-1]]) #append new action had been done into actions
    return temp #list actions to get success
    
def cost(actions):
    """A cost function"""
    return actions.count('l')+actions.count('r')+actions.count('u')+actions.count('d') #return total amount of actions
    #return len([x for x in actions if x.islower()])

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState) #beginBox contains all position of boxes
    beginPlayer = PosOfPlayer(gameState) #beginPlayer contains position of player

    startState = (beginPlayer, beginBox) #startState contains the initial position of boxes and player
    frontier = PriorityQueue() #initiate a variable "frontier" is a priority queue
    frontier.push([startState], 0) #push value of startState with priority cost into frontier
    exploredSet = set()  #initiate a variable "exploredSet" to map all environment and check steps exited or not then skip or add
    actions = PriorityQueue() #initiate a variable "actions" is a priority queue
    actions.push([0], 0) #push [0] with priority cost into frontier
    temp = [] #initiate an empty temp
    ### Implement uniform cost search here
    while frontier:
        node = frontier.pop() #node is the smallest element popped 
        node_action = actions.pop() #node_action is the smallest element popped 
        if isEndState(node[-1][-1]): #check if the solution found out (boxes are at goal positions)
            temp += node_action[1:] #return a list of actions that completed games
            break
        if node[-1] not in exploredSet: #if found node still not checked yet
            exploredSet.add(node[-1]) #add to a variable "exploredSet"
            for action in legalActions(node[-1][0], node[-1][1]): #initiate an iteration with all posible actions of current posPlayer and posBox
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) #update new state for posPlayer and posBox
                if isFailed(newPosBox):  #if new state is invalid skip
                    continue
                #temp = node_action + [action[-1]]
                frontier.push(node + [(newPosPlayer, newPosBox)],cost(node_action + [action[-1]])) #push new position of box and player with a priority cost into frontier
                actions.push(node_action + [action[-1]],cost(node_action + [action[-1]])) #append new action had been done with a priority cost into actions
    return temp #list actions to get success

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':
        result = breadthFirstSearch(gameState)    
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)
    return result

# Monte Carlo Tree Search (MCTS) Tutorial in R
#
# Game Description:
#   - State: a scalar number (starting at 0).
#   - Actions: add 1 or add 2.
#   - Terminal condition: state >= 10.
#         * If state == 10, reward = +1 (win).
#         * If state > 10, reward = –1 (loss).
#
# The algorithm performs:
#   1. Selection using UCT.
#   2. Expansion of an untried action.
#   3. Random simulation (roll–out) until terminal state.
#   4. Backpropagation of the simulation reward.
#
# The best action from the root is selected based on the highest average reward.

set.seed(1)

# Game Parameters
target <- 10         # Terminal state threshold
actions <- c(1, 2)   # Available actions

# MCTS Parameters
numIterations <- 1000  # Total MCTS iterations
uctConstant <- 1.41    # Exploration constant for UCT

# Define functions for MCTS
createNode <- function(state, parent, actions, actionFromParent) {
  list(
    state = state,
    parent = parent,           # Use NA for root
    children = c(),
    visits = 0,
    totalReward = 0,
    untriedActions = actions,  # All actions available initially
    actionFromParent = actionFromParent
  )
}

selectChildUCT <- function(tree, parentIdx, uctConst) {
  children <- tree[[parentIdx]]$children
  bestUCT <- -Inf
  bestChild <- children[1]
  for (childIdx in children) {
    if (tree[[childIdx]]$visits == 0) {
      uctValue <- Inf
    } else {
      avgReward <- tree[[childIdx]]$totalReward / tree[[childIdx]]$visits
      uctValue <- avgReward + uctConst * sqrt(log(tree[[parentIdx]]$visits + 1) / tree[[childIdx]]$visits)
    }
    if (uctValue > bestUCT) {
      bestUCT <- uctValue
      bestChild <- childIdx
    }
  }
  bestChild
}

isTerminal <- function(state, target) {
  state >= target
}

nextState <- function(state, action) {
  state + action
}

rollout <- function(state, target, actions) {
  while (!isTerminal(state, target)) {
    state <- nextState(state, sample(actions, 1))
  }
  if (state == target) 1 else -1
}

# Initialize tree as a list of nodes
tree <- list()
tree[[1]] <- createNode(0, NA, actions, NA)  # Root node
nodeCount <- 1

# Run MCTS iterations
for (iter in 1:numIterations) {
  current <- 1  # Start at root
  # Selection: traverse until a node with untried actions or a terminal node is reached
  while (!isTerminal(tree[[current]]$state, target) &&
         length(tree[[current]]$untriedActions) == 0 &&
         length(tree[[current]]$children) > 0) {
    current <- selectChildUCT(tree, current, uctConstant)
  }
  
  # Expansion
  if (!isTerminal(tree[[current]]$state, target) &&
      length(tree[[current]]$untriedActions) > 0) {
    action <- tree[[current]]$untriedActions[1]
    tree[[current]]$untriedActions <- tree[[current]]$untriedActions[-1]
    newState <- nextState(tree[[current]]$state, action)
    nodeCount <- nodeCount + 1
    newNode <- createNode(newState, current, actions, action)
    tree[[nodeCount]] <- newNode
    tree[[current]]$children <- c(tree[[current]]$children, nodeCount)
    current <- nodeCount
  }
  
  # Simulation (roll–out)
  reward <- rollout(tree[[current]]$state, target, actions)
  
  # Backpropagation
  idx <- current
  while (!is.na(idx)) {
    tree[[idx]]$visits <- tree[[idx]]$visits + 1
    tree[[idx]]$totalReward <- tree[[idx]]$totalReward + reward
    idx <- tree[[idx]]$parent
  }
}

# Choose the best action from the root based on average reward
bestAvg <- -Inf; bestChild <- NA
for (childIdx in tree[[1]]$children) {
  avgReward <- tree[[childIdx]]$totalReward / tree[[childIdx]]$visits
  if (avgReward > bestAvg) {
    bestAvg <- avgReward
    bestChild <- childIdx
  }
}
bestAction <- tree[[bestChild]]$actionFromParent
cat(sprintf("From state %d, the best action is: +%d (avg reward: %.2f)\n", tree[[1]]$state, bestAction, bestAvg))

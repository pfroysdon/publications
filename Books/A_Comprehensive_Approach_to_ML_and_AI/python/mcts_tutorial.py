import numpy as np
import math
import random

def create_node(state, parent, actions, action_from_parent):
    return {
        'state': state,
        'parent': parent,           # 0 for root
        'children': [],
        'visits': 0,
        'total_reward': 0,
        'untried_actions': actions.copy(),
        'action_from_parent': action_from_parent
    }

def select_child_uct(tree, parent_idx, uct_const):
    children = tree[parent_idx]['children']
    best_uct = -np.inf
    best_child = children[0]
    for child_idx in children:
        child = tree[child_idx]
        if child['visits'] == 0:
            uct_value = np.inf
        else:
            avg_reward = child['total_reward'] / child['visits']
            uct_value = avg_reward + uct_const * math.sqrt(math.log(tree[parent_idx]['visits'] + 1) / child['visits'])
        if uct_value > best_uct:
            best_uct = uct_value
            best_child = child_idx
    return best_child

def is_terminal(state, target):
    return state >= target

def next_state(state, action):
    return state + action

def rollout(state, target, actions):
    while not is_terminal(state, target):
        a = random.choice(actions)
        state = next_state(state, a)
    return 1 if state == target else -1

if __name__ == '__main__':
    target = 10
    actions = [1, 2]
    num_iterations = 1000
    uct_constant = 1.41

    tree = []
    tree.append(create_node(state=0, parent=0, actions=actions, action_from_parent=None))  # root node, index 0
    
    for _ in range(num_iterations):
        # SELECTION
        current = 0
        while (not is_terminal(tree[current]['state'], target) and 
               len(tree[current]['untried_actions']) == 0 and 
               len(tree[current]['children']) > 0):
            current = select_child_uct(tree, current, uct_constant)
        
        # EXPANSION
        if not is_terminal(tree[current]['state'], target) and tree[current]['untried_actions']:
            action = tree[current]['untried_actions'].pop(0)
            new_state = next_state(tree[current]['state'], action)
            new_node = create_node(new_state, parent=current, actions=actions, action_from_parent=action)
            tree.append(new_node)
            new_idx = len(tree) - 1
            tree[current]['children'].append(new_idx)
            current = new_idx
        
        # SIMULATION (Rollout)
        reward = rollout(tree[current]['state'], target, actions)
        
        # BACKPROPAGATION
        idx = current
        while True:
            tree[idx]['visits'] += 1
            tree[idx]['total_reward'] += reward
            if idx == tree[idx]['parent']:
                break
            idx = tree[idx]['parent']
            if idx == 0:
                # Also update the root
                tree[0]['visits'] += 0  # already updated in loop
                tree[0]['total_reward'] += 0  # no change needed
                break
    
    # Choose best action from root
    best_avg = -np.inf
    best_child = None
    for child_idx in tree[0]['children']:
        child = tree[child_idx]
        avg_reward = child['total_reward'] / child['visits']
        if avg_reward > best_avg:
            best_avg = avg_reward
            best_child = child_idx
    best_action = tree[best_child]['action_from_parent']
    print(f"From state {tree[0]['state']}, the best action is: +{best_action} (avg reward: {best_avg:.2f})")

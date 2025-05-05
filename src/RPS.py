# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import numpy as np

ALPHA, GAMMA, EPSILON = 0.4, 0.9, 0.01
STATES, ACTIONS = 40, 3
Q = np.zeros((STATES, ACTIONS))

states, state, guess, state_idx = [], '', None, None

def counter_move(move):
    winners = {'R': 'P', 'P': 'S', 'S': 'R'}
    return winners[move]

def update_state(state, move, window_size=3):
    state += move
    if len(state) > window_size:
        state = state[1:]
    return state

def get_idx_state(state, states):
    if state not in states:
        states.append(state)
    return states.index(state)

def player(prev_play, opponent_history=[]):
    global STATES, ACTIONS, Q, ALPHA, GAMMA, EPSILON, state, state_idx, guess, states

    options = ['R', 'P', 'S']

    if prev_play == '':
        Q = np.zeros((STATES, ACTIONS))
        guess, EPSILON, states, state, state_idx = None, 0.01, [], '', None

    if guess:
        if (guess == "P" and prev_play == "R") or (
            guess == "R" and prev_play == "S") or (
            guess == "S" and prev_play == "P"):
            reward = 1
        else:
            reward = -1

        guess_action = options.index(guess)

        next_state = update_state(state, prev_play)
        next_state_idx = get_idx_state(next_state, states)

        Q[state_idx, guess_action] = Q[state_idx, guess_action] + ALPHA * (reward + GAMMA * np.max(Q[next_state_idx, :]) - Q[state_idx, guess_action])

        state = next_state

        if reward == 1:
            EPSILON -= 0.009

    state_idx = get_idx_state(state, states)

    if np.random.uniform(0, 1) < EPSILON:
        action = np.random.choice(ACTIONS)
    else:
        action = np.argmin(Q[state_idx, :])

    prediction = options[action]
    guess = counter_move(counter_move(prediction))

    return guess
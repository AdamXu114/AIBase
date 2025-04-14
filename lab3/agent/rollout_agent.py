import copy
import random
from joblib import Parallel, delayed

from agent.base_agent import BaseAgent, raiseNotDefined

def rollout_once(game, action):
    game_sim = copy.deepcopy(game)
    game_sim.set_state(game.get_state())
    game_sim.set_action(action)
    game_sim.forward()
    while True:
        over, state_val, total_score = game_sim.is_game_over()
        if over:
            return total_score
        else:
            possible_actions_sim = game_sim.get_valid_actions()
            if possible_actions_sim:
                action_sim = random.choice(possible_actions_sim)
                game_sim.set_action(action_sim)
                game_sim.forward()

class RolloutAgent(BaseAgent):
    def __init__(self, game, ui, num_rollouts=30):
        super().__init__(game, ui)
        self._num_rollouts = num_rollouts
       
    def _get_action(self):
        current_state = self._game.get_state()
        actions = self._game.get_valid_actions()
        utility_actions = {}
        for action in actions:
            mean_score = 0
            game_sim = copy.deepcopy(self._game)
            for num_iter in range(self._num_rollouts):
                game_sim.set_state(current_state.copy())
                mean_score += 1/(num_iter+1) * rollout_once(game_sim, action)
            utility_actions[action] = mean_score
        action_todo, _ = max(utility_actions.items(), key=lambda k: k[1])
        return action_todo
        # raiseNotDefined()

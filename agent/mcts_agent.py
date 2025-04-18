import random

from agent.base_agent import raiseNotDefined, BaseAgent
import copy
from collections import defaultdict
import math

from game.game import Game2048

from agent.heuristic import exp_heuristic2 as heuristic
def rollout_once(game, action, max_depth = 50):
    game_sim = copy.deepcopy(game)
    game_sim.set_state(game.get_state())
    game_sim.set_action(action)
    game_sim.forward()
    while True:
        over, state_val, total_score = game_sim.is_game_over()
        if over:
            # print("heuristic_value:", heuristic_val)
            return total_score if not (state_val == game.max_value()) else 1e7
        if max_depth==0:
            heuristic_val = heuristic(game.get_state())
            return heuristic_val if not (state_val == game.max_value()) else 1e7
        else:
            max_depth -= 1
            possible_actions_sim = game_sim.get_valid_actions()
            if possible_actions_sim:
                action_sim = random.choice(possible_actions_sim)
                game_sim.set_action(action_sim)
                game_sim.forward()


class MCTSAgent(BaseAgent):
    def __init__(self, game, ui, num_rollouts=100):
        super().__init__(game, ui)
        self.num_rollouts = num_rollouts
        self.num_simulations = defaultdict(lambda: defaultdict(int))
        self.state_info = defaultdict(lambda : [0.0, 0])    # value, num


    def _get_action(self):
        # self.num_simulations.clear()
        # self.state_info.clear()
        actions = self._game.get_valid_actions()
        utility_actions = {}
        current_state = self._game.get_state()
        for num_iter in range(self.num_rollouts):
            self._simulate_once(current_state)
        for action in actions:
            game_sim = copy.deepcopy(self._game)
            game_sim.set_state(current_state)
            game_sim.set_action(action)
            game_sim.forward()
            next_state_key = tuple(tuple(row) for row in game_sim.get_state())
            # 修改获取方式
            utility_actions[action] = self.state_info.get(next_state_key, [0.0])[0]
        action_todo, _ = max(utility_actions.items(), key=lambda k: k[1])
        return action_todo
        # raiseNotDefined()

    def _ucb(self, gameState, action):
        state_key = tuple(tuple(row) for row in gameState)
        if self.num_simulations[state_key][action] == 0:
            return float('inf')
        state_visits = self.state_info[state_key][1]
        action_visits = self.num_simulations[state_key][action]
        return (self.state_info[state_key][0] +
                2 * math.sqrt(math.log2(state_visits)/action_visits))

    def _select_action(self, gameState):
        state_key = tuple(tuple(row) for row in gameState)
        game_sim = copy.deepcopy(self._game)
        game_sim.set_state(gameState)
        actions = game_sim.get_valid_actions()
        utility_actions = {action : self._ucb(gameState,action)for action in actions}
        action_todo, _ = max(utility_actions.items(), key=lambda k: k[1])
        return action_todo

    def _simulate_once(self, gameState):
        state_key = tuple(tuple(row) for row in gameState)
        if not state_key in self.state_info:
            game_sim = copy.deepcopy(self._game)
            game_sim.set_state(gameState)
            # 新增：检查游戏是否已结束
            over, state_val, total_score = game_sim.is_game_over()
            if over:
                heuristic_val = heuristic(gameState)
                # print("heuristic_value:", heuristic_val)
                return total_score + heuristic_val if not (state_val == self._game.max_value()) else float('inf')
            possible_actions_sim = game_sim.get_valid_actions()
            for action in possible_actions_sim:
                self.num_simulations[state_key][action] = 0  # 初始为0但返回inf

            action_sim = random.choice(possible_actions_sim)
            score = rollout_once(game_sim,action_sim)

            self.state_info[state_key] = [score,1]
            self.num_simulations[state_key][action_sim] = 1
            return score


        selected_action = self._select_action(gameState)
        game_sim = copy.deepcopy(self._game)
        game_sim.set_state(gameState)
        game_sim.set_action(selected_action)
        game_sim.forward()
        score_fdb = self._simulate_once(game_sim.get_state())
        self.state_info[state_key] = [(self.state_info[state_key][0]
                                         + (score_fdb-self.state_info[state_key][0])/(1+self.state_info[state_key][1]))
                                         ,self.state_info[state_key][1] + 1]
        self.num_simulations[state_key][selected_action] = 1 + self.num_simulations[state_key][selected_action]
        return self.state_info[state_key][0]


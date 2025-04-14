import copy
from agent.base_agent import BaseAgent, raiseNotDefined
from agent.heuristic import exp_heuristic


class ExpectimaxAgent(BaseAgent):
    def __init__(self, game,ui, max_depth=5,heuristic=exp_heuristic):
        super().__init__(game, ui)
        self._max_depth = max_depth
        self._heuristic = heuristic
        self._cache = {}


    def expectimax(self, gameState, is_max_player, depth):
        game_branch = copy.deepcopy(self._game)
        game_branch.set_state(gameState)
        if game_branch.is_game_over()[0]:
            return float('inf') if game_branch.is_won() else -float('inf')
        if depth <= 0:
                return self._heuristic(gameState)
        if is_max_player == False:
            depth -= 1

        if is_max_player == True:
            return self.max_value(gameState, depth)
        else:
            return self.exp_value(gameState,  depth)
        
    def max_value(self, gameState, depth):
        game_branch = copy.deepcopy(self._game)
        game_branch.set_state(gameState)
        v = -float('inf')
        for action in game_branch.get_valid_actions():
            branch_game2 = copy.deepcopy(game_branch)  
            branch_game2.set_action(action)
            branch_game2.forward_player_only()
            next_state = branch_game2.get_state()
            v = max(v, self.expectimax(next_state, False, depth))
        return v

    def exp_value(self, gameState, depth):
        game_branch = copy.deepcopy(self._game)
        game_branch.set_state(gameState)
        v = 0
        for new_state, prob in game_branch.get_valid_successors():
            v += prob * self.expectimax(new_state, True, depth)  # 完成完整回合后递减深度
        return v    
    
    def _get_action(self):

        current_state = self._game.get_state()
        actions = self._game.get_valid_actions()
        utility_actions = {}
        for action in actions:
            game_sim = copy.deepcopy(self._game)
            game_sim.set_state(current_state)
            game_sim.set_action(action)
            game_sim.forward_player_only()
            next_state = game_sim.get_state()
            utility_actions[action] = self.expectimax(next_state, self._max_depth, False)
        action_todo, _ = max(utility_actions.items(), key=lambda k: k[1])
        return action_todo
    
    
    
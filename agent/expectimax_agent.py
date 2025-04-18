import copy
from agent.base_agent import BaseAgent, raiseNotDefined
from agent.heuristic import exp_heuristic


class ExpectimaxAgent(BaseAgent):
    def __init__(self, game,ui, max_depth=5,heuristic=exp_heuristic):
        super().__init__(game, ui)
        self._max_depth = max_depth
        self._heuristic = heuristic
        
    def _get_action(self):
        raiseNotDefined()

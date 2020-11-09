from board import Board
from search import SearchProblem, ucs, astar
import util
import numpy as np
import math

class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)



#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.board = Board(self.board_w, self.board_h, 1, self.piece_list, starting_point)
        self.expanded = 0
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        return state.get_position(0, 0) == 0 and \
               state.get_position(self.board_w - 1, 0) == 0 and \
               state.get_position(0, self.board_h - 1) == 0 and \
               state.get_position(self.board_w - 1, self.board_h - 1) == 0

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        return sum([action.piece.get_num_tiles() for action in actions])


def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    "*** YOUR CODE HERE ***"
    # TODO Heuristic doesn't reach zero when we reach goal
    # corner_cost = np.count_nonzero([state.get_position(0, 0),
    #                                 state.get_position(problem.board_w - 1, 0),
    #                                 state.get_position(0, problem.board_h - 1),
    #                                 state.get_position(problem.board_w - 1, problem.board_h - 1)])
    # cost0 = np.count_nonzero(np.sum(state.state[:, 1:-1] + 1, axis=0))
    # cost1 = np.count_nonzero(np.sum(state.state[1:-1, :] + 1, axis=1))
    # return max(cost0, cost1) + corner_cost
    corner1 = 0 if state.get_position(0, 0) == 0 else np.inf
    corner2 = 0 if state.get_position(0, state.board_h - 1) == 0 else np.inf
    corner3 = 0 if state.get_position(state.board_w - 1, 0) == 0 else np.inf
    corner4 = 0 if state.get_position(state.board_w - 1, state.board_h - 1) == 0 else np.inf

    if (not state.check_tile_legal(0, 0, 0) and corner1 != 0) or \
            (not state.check_tile_legal(0, 0, state.board_h - 1) and corner2 != 0) or \
            (not state.check_tile_legal(0, state.board_w - 1, 0) and corner3 != 0) or \
            (not state.check_tile_legal(0, state.board_w - 1, state.board_h - 1) and corner4 != 0):
        return np.inf  # no reason to continue with this state!

    for h in range(state.board_h):
        for w in range(state.board_w):
            if state.check_tile_legal(0, w, h) and state.connected[0][h][w]:
                corner1 = min(corner1, max(h + 1, w + 1))
                corner2 = min(corner2, max(state.board_h - h, w + 1))
                corner3 = min(corner3, max(h + 1, state.board_w - w))
                corner4 = min(corner4, max(state.board_h - h, state.board_w - w))

    return max(corner1, corner2, corner3, corner4)


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.board = Board(self.board_w, self.board_h, 1, self.piece_list, starting_point)
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        # TODO check if I have to check for special situations

        for i in self.targets:
            if state.get_position(i[1], i[0]) != 0:
                return False
        return True



    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        return sum([action.piece.get_num_tiles() for action in actions])


def blokus_cover_heuristic(state, problem):
    "*** YOUR CODE HERE ***"
    target = problem.targets
    count_all = 0
    for t in target:
        min_dist = problem.board_w * problem.board_h
        for iy, ix in np.ndindex(state.state.shape):
            if state.state[iy, ix] == 0:
                dist = (t[0] - iy)**2 + (t[1] - ix)**2
                dist = math.sqrt(dist) / 2

                if dist < min_dist:
                    min_dist = dist
        count_all = count_all + min_dist
    return count_all


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.expanded = 0
        self.targets = targets.copy()
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.starting_point = starting_point
        self.board = Board(self.board_w, self.board_h, 1, self.piece_list, starting_point)
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest
        uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []

        while ....

            actions = set of actions that covers the closets uncovered target location
            add actions to backtrace

        return backtrace
        """
        "*** YOUR CODE HERE ***"
        current_state = self.board.__copy__()

        backtrace = []
        small_list = [targ for targ in self.targets]
        while small_list:
            target = findClosestTarget(current_state, small_list)
            small_list.remove(target)
            problem = BlokusCoverProblem(current_state.board_w, current_state.board_h,
                                         current_state.piece_list, self.starting_point,
                                         [target])
            problem.board = current_state
            moves = astar(problem, blokus_cover_heuristic)
            for move in moves:
                current_state = current_state.do_move(0, move)
                backtrace.append(move)

            self.expanded += problem.expanded
        return backtrace

def findClosestTarget(state, targets):
    min_dist = state.board_w * state.board_h  # float('inf')
    t = targets[0]
    for target in targets:
        for x in range(state.board_w):
            for y in range(state.board_h):
                if state.check_tile_legal(0, x, y) and state.connected[0][y][x]:
                    dist = math.sqrt((target[1] - x) ** 2 + (target[0] - y) ** 2) / 2
                    if dist < min_dist:
                        min_dist = dist
                        t = target
    return t


class Items:

    def __init__(self, state, actions, total_cost):
        self.state = state
        self.actions = actions
        self.total_cost = total_cost

class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self):
        """
        check how to make it viable since I need to check for one state only at a time
        and not get confused with previously achieved goals
        :param state:
        :return:
        """
        pass

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        pass

    def solve(self):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()







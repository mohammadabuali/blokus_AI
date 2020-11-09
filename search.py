"""
In search.py, you will implement generic search algorithms
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()




def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state().state)
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.get_start_state().state)
    # print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    # print("Start's successors:", len(problem.get_successors(problem.get_start_state())))
    # s = problem.get_successors(problem.get_start_state())[0]
    # print("Start's successors:", s)
    # print("Start's successors:", s[0])
    # print("Start's successors:", s[1])
    # print("Start's successors:", s[2])

    stack = util.Stack()
    start = problem.get_start_state()
    stack.push((start, []))
    visited = set()

    while not stack.isEmpty():
        state, actions = stack.pop()
        if problem.is_goal_state(state):
            return actions

        if state not in visited:
            visited.add(state)
            successors = problem.get_successors(state)
            for successor, action, _ in successors:
                stack.push((successor, actions + [action]))


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"

    queue = util.Queue()
    start = problem.get_start_state()
    queue.push((start, []))
    visited = set()

    while not queue.isEmpty():
        state, actions = queue.pop()
        if problem.is_goal_state(state):
            return actions

        if state not in visited:
            visited.add(state)
            successors = problem.get_successors(state)
            for successor, action, _ in successors:
                queue.push((successor, actions + [action]))


class Item:

    def __init__(self, state, actions, total_cost):
        self.state = state
        self.actions = actions
        self.total_cost = total_cost


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"
    priority_queue = util.PriorityQueueWithFunction(lambda x: x.total_cost)
    start = problem.get_start_state()
    priority_queue.push(Item(start, [], 0))
    visited = set()

    while not priority_queue.isEmpty():
        item = priority_queue.pop()
        if problem.is_goal_state(item.state):
            return item.actions

        if item.state not in visited:
            visited.add(item.state)
            successors = problem.get_successors(item.state)
            for successor, action, step_cost in successors:
                priority_queue.push(Item(successor, item.actions + [action], item.total_cost + step_cost))


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    "*** YOUR CODE HERE ***"
    priority_queue = util.PriorityQueueWithFunction(lambda x: x.total_cost + heuristic(x.state, problem))
    start = problem.get_start_state()
    priority_queue.push(Item(start, [], 0))
    visited = set()

    while not priority_queue.isEmpty():
        item = priority_queue.pop()
        if problem.is_goal_state(item.state):
            return item.actions

        if item.state not in visited:
            visited.add(item.state)
            successors = problem.get_successors(item.state)
            for successor, action, step_cost in successors:
                priority_queue.push(Item(successor, item.actions + [action], item.total_cost + step_cost))



# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search

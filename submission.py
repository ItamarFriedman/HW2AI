from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    # distance form the closest package if robot has no package
    pickUpIncentive = (robot.package is None) * min(manhattan_distance(robot.position, env.packages[0].position),
                                                    manhattan_distance(robot.position, env.packages[1].position))
    # finding the closest charge
    closestCharge = env.charge_stations[1]
    if manhattan_distance(robot.position, env.charge_stations[0].position) <= \
            manhattan_distance(robot.position, env.charge_stations[1].position):
        closestCharge = env.charge_stations[0]

    closestPackageToCharge = env.packages[1]
    if manhattan_distance(closestCharge.position, env.packages[0].position) <= \
            manhattan_distance(closestCharge.position, env.packages[1].position):
        closestPackageToCharge = env.packages[0]

    after_charge = manhattan_distance(closestCharge.position, closestPackageToCharge.position) + \
                   manhattan_distance(closestPackageToCharge.position, closestPackageToCharge.destination)

    # if robot does have a package, setting his drop-off distance and extra distance from charge to drop-off
    dropOffIncentive = 0
    if robot.package is not None:
        dropOffIncentive = manhattan_distance(robot.position, robot.package.destination)
        after_charge = manhattan_distance(closestCharge.position, robot.package.destination)

    # if battery is low
    if robot.battery <= 1.5 * manhattan_distance(robot.position, closestCharge.position) and \
            (after_charge >= robot.battery + robot.credit):
        return -manhattan_distance(robot.position, closestCharge.position) + 50 * (robot.package is not None)
    else:  # if battery is high enough
        return -(15 * pickUpIncentive + dropOffIncentive - (100 * robot.battery) *
                 (after_charge >= robot.battery + robot.credit) -
                 50 * robot.credit) + 50 * (robot.package is not None) + robot.position[1]


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start = time.time()
        depth = 0
        o = 'park'
        try:
            while True:
                depth += 1
                h, o = self.rb_minimax(env, agent_id, depth, True, start, time_limit)
                if o == -1:
                    return 'park'
        except TimeoutError:
            return o

    def rb_minimax(self, env, agent_id, depth, our_turn, start, time_limit) -> (int, int):
        currentTime = time.time()
        if currentTime - start >= 0.9 * time_limit:
            raise TimeoutError
        if (env.done()) or depth == 0:
            return self.heuristic(env, agent_id), -1

        if our_turn:
            current_agent_id = agent_id
        else:
            current_agent_id = not agent_id
        operators = env.get_legal_operators(current_agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(current_agent_id, op)
        if our_turn:
            current_max = -float('inf')
            op_max = -1
            for child, op in zip(children, operators):
                v, _ = self.rb_minimax(child, agent_id, depth - 1, not our_turn, start, time_limit)
                if v > current_max:
                    current_max = v
                    op_max = op
            return current_max, op_max
        else:
            current_min = float('inf')
            op_min = -1
            for child, op in zip(children, operators):
                v, _ = self.rb_minimax(child, agent_id, depth - 1, not our_turn, start, time_limit)
                if v < current_min:
                    current_min = v
                    op_min = op
            return current_min, op_min

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        robot = env.get_robot(robot_id)
        other_robot = env.get_robot((robot_id + 1) % 2)

        if (env.done()):
            return 10e6 * (robot.credit > other_robot.credit) - 10e6 * (robot.credit < other_robot.credit)
        return smart_heuristic(env, robot_id) - smart_heuristic(env, not robot_id)

        packageBonus = 1 * (robot.package is not None)
        creditBonus = 10 * robot.credit
        lowBatteryPanelty = 500 * (robot.battery <= 0)
        creditDifference = 100 * (robot.credit - other_robot.credit)
        ourHeuristic = packageBonus + creditBonus + creditDifference - lowBatteryPanelty

        packageBonus = 1 * (other_robot.package is not None)
        creditBonus = 10 * other_robot.credit
        lowBatteryPanelty = 500 * (other_robot.battery <= 0)
        creditDifference = 100 * (other_robot.credit - robot.credit)
        otherHeuristic = packageBonus + creditBonus + creditDifference - lowBatteryPanelty
        return ourHeuristic - otherHeuristic


class AgentAlphaBeta(AgentMinimax):
    # TODO: section c : 1
    def rb_minimax(self, env, agent_id, depth, our_turn, start, time_limit, alpha=-float('inf'), beta=float('inf')) -> (
            int, int):
        currentTime = time.time()
        if currentTime - start >= 0.9 * time_limit:
            raise TimeoutError
        if (env.done()) or depth == 0:
            return self.heuristic(env, agent_id), -1

        if our_turn:
            current_agent_id = agent_id
        else:
            current_agent_id = not agent_id
        operators = env.get_legal_operators(current_agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(current_agent_id, op)
        if our_turn:
            current_max = -float('inf')
            op_max = -1
            for child, op in zip(children, operators):
                v = self.rb_minimax(child, agent_id, depth - 1, not our_turn, start, time_limit, alpha, beta)[0]
                if v > current_max:
                    current_max = v
                    op_max = op
                alpha = max(current_max, alpha)
                if current_max >= beta:
                    return float('inf'), -1
            return current_max, op_max
        else:
            current_min = float('inf')
            op_min = -1
            for child, op in zip(children, operators):
                v = self.rb_minimax(child, agent_id, depth - 1, not our_turn, start, time_limit, alpha, beta)[0]
                if v < current_min:
                    current_min = v
                    op_min = op
                beta = min(current_min, beta)
                if current_min <= alpha:
                    return -float('inf'), -1
            return current_min, op_min


class AgentExpectimax(AgentMinimax):
    # TODO: section d : 1
    def rb_minimax(self, env, agent_id, depth, our_turn, start, time_limit) -> (int, int):
        currentTime = time.time()
        if currentTime - start >= 0.9 * time_limit:
            raise TimeoutError
        if (env.done()) or depth == 0:
            return self.heuristic(env, agent_id), -1
        if our_turn:
            current_agent_id = agent_id
        else:
            current_agent_id = not agent_id
        operators = env.get_legal_operators(current_agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(current_agent_id, op)
        if our_turn:
            current_max = -float('inf')
            op_max = -1
            for child, op in zip(children, operators):
                v, _ = self.rb_minimax(child, agent_id, depth - 1, not our_turn, start, time_limit)
                if v > current_max:
                    current_max = v
                    op_max = op
            return current_max, op_max
        else:
            sum = 0
            for child, op in zip(children, operators):
                v, _ = self.rb_minimax(child, agent_id, depth - 1, not our_turn, start, time_limit)
                sum += v * (op == "move east" or op == "pick up") + v
            sum /= len(operators) + ("move east" in operators) + ("pick up" in operators)
            return sum, -1


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)

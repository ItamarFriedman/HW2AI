from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    pickUpIncentive = (robot.package is None) * min(manhattan_distance(robot.position, env.packages[0].position),
                                                    manhattan_distance(robot.position, env.packages[1].position))
    closestCharge = env.charge_stations[1]
    if manhattan_distance(robot.position, env.charge_stations[0].position) <= manhattan_distance(robot.position,
                                                                                                 env.charge_stations[
                                                                                                     1].position):
        closestCharge = env.charge_stations[0]

    extra = min(manhattan_distance(closestCharge.position, env.packages[0].position),
                manhattan_distance(closestCharge.position, env.packages[1].position))

    dropOffIncentive = 0
    if robot.package is not None:
        dropOffIncentive = manhattan_distance(robot.position, robot.package.destination)
        extra = manhattan_distance(closestCharge.position, robot.package.destination)

    if robot.battery <= 1.5 * min(manhattan_distance(robot.position, env.charge_stations[0].position),
                            manhattan_distance(robot.position, env.charge_stations[1].position)):

        return 50000 - min(manhattan_distance(robot.position, env.charge_stations[0].position),
                           manhattan_distance(robot.position, env.charge_stations[1].position))
    else:
        return 50000 - (15 * pickUpIncentive + dropOffIncentive - (100 * robot.battery) * (extra >= robot.battery) -
                        50*robot.credit)





class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start = time.time()
        operators = env.get_legal_operators(agent_id)
        depth = 0
        h, o = (0, 0)
        try:
            while True:
                depth += 1
                h, o = self.rb_minimax(env, agent_id, depth, True, start, time_limit)
                #print("Operation ", o, "Hueristic", h)
                #print(depth)
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
                v, _ = self.rb_minimax(child, agent_id, depth-1, not our_turn, start, time_limit)
                if v > current_max :
                    current_max = v
                    op_max = op
            return current_max, op_max
        else:
            current_min = float('inf')
            op_min = -1
            for child, op in zip(children, operators):
                v, _ = self.rb_minimax(child, agent_id, depth-1, not our_turn, start, time_limit)
                if v < current_min:
                    current_min = v
                    op_min = op
            return current_min, op_min

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        robot = env.get_robot(robot_id)
        other_robot = env.get_robot((robot_id + 1) % 2)
        packageBonus = 10*(robot.package is not None)
        creditBonus = 100*robot.credit
        creditDifference = 1000*(robot.credit - other_robot.credit)
        return packageBonus + creditBonus + creditDifference

class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


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

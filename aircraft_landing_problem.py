import os
import random
from qubots.base_problem import BaseProblem

class AircraftLandingProblem(BaseProblem):
    """
    Aircraft Landing Problem:
    Given an instance of the aircraft landing scheduling problem, schedule landing times
    and a landing order for a set of planes so as to minimize the total penalty cost
    (due to early or late landings) while respecting separation time constraints.
    
    Instance data (from the OR-Library) has the following format:
      - First line: number of planes and freeze time.
      - For each plane:
          appearance time (ignored),
          earliest landing time,
          target landing time,
          latest landing time,
          penalty cost per unit time for landing early,
          penalty cost per unit time for landing late,
          then for each plane j: separation time required after this plane lands before j can land.
      
    Note: The static version ignores appearance time and freeze time.
    """

    def __init__(self, instance_file=None, nb_planes=None, earliest_time=None, target_time=None,
                 latest_time=None, earliness_cost=None, tardiness_cost=None, separation_time=None):
        if instance_file is not None:
            self._load_instance_from_file(instance_file)
        else:
            # Data provided directly (for testing or alternative use)
            if (nb_planes is None or earliest_time is None or target_time is None or
                latest_time is None or earliness_cost is None or tardiness_cost is None or
                separation_time is None):
                raise ValueError("Either 'instance_file' or all instance arrays must be provided.")
            self.nb_planes = nb_planes
            self.earliest_time = earliest_time
            self.target_time = target_time
            self.latest_time = latest_time
            self.earliness_cost = earliness_cost
            self.tardiness_cost = tardiness_cost
            self.separation_time = separation_time

    def _load_instance_from_file(self, filename):
        # Resolve relative paths with respect to this module’s directory
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)
        with open(filename, "r") as f:
            tokens = f.read().split()
        it = iter(tokens)
        self.nb_planes = int(next(it))
        _ = next(it)  # skip freeze time (unused)
        self.earliest_time = []
        self.target_time = []
        self.latest_time = []
        self.earliness_cost = []
        self.tardiness_cost = []
        self.separation_time = []
        for p in range(self.nb_planes):
            _ = next(it)  # skip appearance time
            self.earliest_time.append(int(next(it)))
            self.target_time.append(int(next(it)))
            self.latest_time.append(int(next(it)))
            self.earliness_cost.append(float(next(it)))
            self.tardiness_cost.append(float(next(it)))
            sep_row = []
            for _ in range(self.nb_planes):
                sep_row.append(int(next(it)))
            self.separation_time.append(sep_row)

    def evaluate_solution(self, solution) -> float:
        """
        Evaluate a candidate solution.
        
        The candidate solution should be a dictionary with keys:
          - "landing_order": a list (permutation) of plane indices.
          - "landing_times": a list of landing times (integers) corresponding to each position.
          
        The function returns the total penalty cost if the solution is feasible.
        If any constraint is violated (e.g., landing time outside its window or separation constraints not met),
        a large penalty is returned.
        """
        PENALTY = 1e9  # large penalty for infeasible solutions
        landing_order = solution.get("landing_order")
        landing_times = solution.get("landing_times")
        if landing_order is None or landing_times is None:
            return PENALTY
        n = self.nb_planes
        if len(landing_order) != n or len(landing_times) != n:
            return PENALTY
        if sorted(landing_order) != list(range(n)):
            return PENALTY

        total_cost = 0.0
        # For each scheduled landing, check time windows and calculate penalty cost.
        for p in range(n):
            plane = landing_order[p]
            lt = landing_times[p]
            if lt < self.earliest_time[plane] or lt > self.latest_time[plane]:
                return PENALTY
            target = self.target_time[plane]
            if lt < target:
                cost = (target - lt) * self.earliness_cost[plane]
            else:
                cost = (lt - target) * self.tardiness_cost[plane]
            total_cost += cost

        # Check separation constraints for consecutive landings.
        for p in range(1, n):
            prev_plane = landing_order[p - 1]
            current_plane = landing_order[p]
            min_sep = self.separation_time[prev_plane][current_plane]
            if landing_times[p] < landing_times[p - 1] + min_sep:
                return PENALTY

        return total_cost

    def random_solution(self):
        """
        Generate a random candidate solution.
        
        The returned solution is a dictionary with:
          - "landing_order": a random permutation of plane indices.
          - "landing_times": a list of landing times computed in a greedy fashion.
          
        For each plane (in the given random order), the landing time is chosen as a random integer
        between the maximum of (its earliest time and the previous landing time plus the required separation)
        and its latest time.
        """
        n = self.nb_planes
        landing_order = list(range(n))
        random.shuffle(landing_order)
        landing_times = []
        for p in range(n):
            plane = landing_order[p]
            if p == 0:
                min_time = self.earliest_time[plane]
            else:
                prev_plane = landing_order[p - 1]
                min_time = max(self.earliest_time[plane],
                               landing_times[p - 1] + self.separation_time[prev_plane][plane])
            max_time = self.latest_time[plane]
            if min_time > max_time:
                landing_time = max_time
            else:
                landing_time = random.randint(min_time, max_time)
            landing_times.append(landing_time)
        return {"landing_order": landing_order, "landing_times": landing_times}

import pulp;
import numpy as np;

def main():
    available_length = np.array([3, 6, 10, 15, 20])
    
    target_length = 10
    
    
    used_lengths = pulp.LpVariable.dicts("used_lengths", available_length, cat=pulp.LpInteger, lowBound = 0)
    
    expr_length = pulp.lpSum([length * used_lengths[length] for length in available_length])
    
    
    # minimize error between target and solved
    # model = pulp.LpProblem("NearestLength", pulp.LpMinimize)
    # error = pulp.LpVariable("error", lowBound=0)
    # model += error
    # model += expr_length - target_length <= error
    # model += expr_length - target_length >= error
    
    # minimize used lengths
    model = pulp.LpProblem("NearestLength", pulp.LpMinimize)
    model += pulp.lpSum([used_lengths[length] for length in available_length])
    model += expr_length == target_length
    # model += expr_length - target_length >= error
    
    # maximize expr_length with constrain to target_length
    # model = pulp.LpProblem("NearestLength", pulp.LpMaximize)
    # model += pulp.lpSum([used_lengths[length] for length in available_length])
    # model += expr_length
    # model += expr_length <= target_length
    
    
    print("Start solving")
    model.solve()

    print("State:", pulp.LpStatus[model.status])

    
    solved_length = sum([length * pulp.value(used_lengths[length]) for length in available_length])    
    
    print(f"Solved length: {solved_length}")
    print(f"Error length(solved - target): {solved_length - target_length}")
    
    print("Length:Count")
    for length in available_length:
        print(f"{length}:{pulp.value(used_lengths[length])}")


if __name__ == "__main__":
    main()

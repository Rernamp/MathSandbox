from pulp import *
import numpy as np
import sys


def main():

    if len(sys.argv) != 2:
        print("need target resistor as argument")
    
    r_total_target = float(sys.argv[1])
    g_total_target = 1 / r_total_target

    target_current = 0.14
    available_resistors = np.array(
        [30, 33, 36, 39, 47, 49.9, 51, 56, 62, 68, 75, 82, 91]
    )
    t_r = 5    

    model = LpProblem("Led_current_limit_resistor", LpMinimize)    

    r_p_used = pulp.LpVariable.dicts("r_p_used", available_resistors, cat="Binary")
    r_s_used = pulp.LpVariable.dicts("r_s_used", available_resistors, cat="Binary")

    model += pulp.lpSum([r_p_used[val] for val in available_resistors]) == 1
    model += pulp.lpSum([r_s_used[val] for val in available_resistors]) == 1

    g_p_calc = pulp.lpSum([r_p_used[r] / r for r in available_resistors])
    g_l_calc = pulp.lpSum([r_s_used[r] / (r + t_r) for r in available_resistors])
    

    error = pulp.LpVariable("error", lowBound=0)

    g_dif = g_p_calc + g_l_calc - g_total_target
    model += (g_dif) <= error
    model += (g_dif) >= error
    model += error
    model.solve()

    print("State:", LpStatus[model.status])    

    r_p_solved = sum(r * pulp.value(r_p_used[r]) for r in available_resistors)
    r_s_solved = sum(r * pulp.value(r_s_used[r]) for r in available_resistors)

    r_total_solved = (r_p_solved * (r_s_solved + t_r)) / (r_p_solved  + r_s_solved + t_r)

    print(f"Target current, A: {target_current}")
    print(f"Available resistors, Om: {np.array2string(available_resistors, separator=",", max_line_width=sys.maxsize)}")
    print(f"Target potentiometer value, Om: {t_r}")
    print(f"Target total resistance {r_total_solved} Om")
    print(f"Solved total resistance {r_total_target} Om")
    print(f"Difference between total and solved resistance {r_total_target - r_total_solved} Om")
    print(f"R_P: {r_p_solved} R_S: {r_s_solved}")

if __name__ == "__main__":
    main()

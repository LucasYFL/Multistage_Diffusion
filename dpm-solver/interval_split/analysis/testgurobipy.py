import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import torch

def solve(nodes, similarity, interval_list, interval_num = 3):
# YOUR MODEL GOES HERE
    m = gp.Model()
    m.reset()
    connect_interval_state = m.addVars(nodes, nodes, interval_num, name="connect_interval_state", vtype=GRB.BINARY)
    interval_state = m.addVars(nodes, interval_num, name="interval_state", vtype=GRB.BINARY)
    interval = m.addVars(nodes, name="interval", vtype=GRB.INTEGER)
    interval_quantity = m.addVars(interval_num, name="1 / (total number for each state)")
    
    m.setObjective(gp.quicksum(similarity[i, j] * connect_interval_state[i, j, k] * interval_quantity[k]  for i in range(nodes) for j in range(nodes) for k in range(interval_num)), GRB.MAXIMIZE)
    # m.setObjective(gp.quicksum(similarity[i, j] * connect_interval_state[i, j, k]  for i in range(nodes) for j in range(nodes) for k in range(interval_num)), GRB.MINIMIZE)


    m.addConstr(interval[0] == 1)
    m.addConstr(interval[nodes - 1] == interval_num)
    m.addConstrs((interval[i + 1] >= interval[i]) for i in range(nodes - 1))
    m.addConstrs((interval[i + 1] <= interval[i] + 1) for i in range(nodes - 1))

    
    m.addConstrs((gp.quicksum(interval_state[i, k] for k in range(interval_num)) == 1) for i in range(nodes))
    m.addConstrs((interval_state[i, k] * (interval[i] - k - 1) == 0) for k in range(interval_num) for i in range(nodes))
    
    m.addConstrs((connect_interval_state[i, j, k] == interval_state[i, k] * interval_state[j, k])  for i in range(nodes) for j in range(nodes) for k in range(interval_num))
    
    m.addConstrs((interval_quantity[k] * gp.quicksum(connect_interval_state[(i, j, k)] for i in range(nodes) for j in range(nodes)) == 1) for k in range(interval_num))
    
    # Specify how to format the output
    # Don't forget: Python indexes from 0!
    def printSolution():
        if m.status == GRB.OPTIMAL:
            print('\nOptimal value: %g' % m.objVal)
            print('\nThe connective graph looks like this:')
            for k in range(interval_num):
                S = ""
                print(f"The {k}th interval")
                for i in range(nodes):
                    for j in range(nodes):
                        S += f"{int(connect_interval_state[(i, j, k)].X)} "    
                    S += "\n"
                print(S) 
            print('\nThe interval status looks like this:')
            S = ""
            for i in range(nodes):
                S += f"{interval[i].X} "    
            S += "\n"
            print(S) 
            print(f"The interval_state looks like this")
            
            for i in range(nodes):
                for k in range(interval_num):
                    S += f"{int(interval_state[(i, k)].X)} "    
                S += "\n"
            print(S) 

            print(f"The interval_quality looks like this")
            for k in range(interval_num):
                S += f"{interval_quantity[(k)].X} "    
            S += "\n"
            print(S)
            i_list = [] 
            for i in range(nodes):
                i_list.append(interval[i].X)
            i_list = np.array(i_list)
            idx = 1
            print(f"The {idx}th is from [0, {interval_list[np.where(i_list == idx)[0].max()] + 0.025})")
            for idx in range(2, interval_num):
                print(f"The {idx}th is from [{interval_list[np.where(i_list == idx)[0].min()] - 0.025}, {interval_list[np.where(i_list == idx)[0].max()] + 0.025})")
            idx = interval_num
            print(f"The {idx}th is from [{interval_list[np.where(i_list == idx)[0].min()] - 0.025}, 1)")
        else:
            print('Optimization ended with status %d' % m.status)
    # Solve
    # To look at the formulation uncomment the line below
    # m.write("out.lp")
    m.setParam('TimeLimit', 5*60)
    m.optimize()
    printSolution()

# nodes=20
# temp=sum(supply[i] for i in range(nodes-1))
# supply=np.append(supply,-1*temp)
# print(sum(supply))
# coords=np.random.rand(nodes,2)
# costs=np.zeros((nodes,nodes))
# for i in range(nodes):
#     for j in range(nodes):
#         costs[i,j]=math.sqrt((coords[i,0]-coords[j,0])**2+(coords[i,1]-
#         coords[j,1])**2)
# # YOUR CODE FOR DETERMINING A VALUE OF BIG M GOES HERE
# # CALCULATE THE VALUE, NAMING IT bigm
# bigm = 1000
# solve(nodes, supply, costs, 10*costs, bigm*np.ones((nodes,nodes)))

nodes=20

similarity=np.random.uniform(0,1, size=(nodes,nodes))
eps = 1e-3
# YOUR CODE FOR DETERMINING A VALUE OF BIG M GOES HERE
# CALCULATE THE VALUE, NAMING IT bigm
solve(nodes, similarity, interval_list = torch.range(eps, 1, 0.05), interval_num = 4)
from itertools import islice
from ortools.algorithms import pywrapknapsack_solver

import time

testgroups = ['n00050', 'n00100' ,'n00200', 'n00500', 'n01000']
groupnames = ['00Uncorrelated', '01WeaklyCorrelated', '02StronglyCorrelated', '03InverseStronglyCorrelated', '04AlmostStronglyCorrelated', '05SubsetSum', '06UncorrelatedWithSimilarWeights', '07SpannerUncorrelated', '08SpannerWeaklyCorrelated', '09SpannerStronglyCorrelated', '10MultipleStronglyCorrelated', '11ProfitCeiling', '12Circle']

for groupname in groupnames:
    for testgroup in testgroups:
        filepath = "d:/OR/" + groupname + '/' #REPLACE ACCORDINGLY

        a = open(filepath + "result.txt", "a+") #output to result.txt

        capacities = []
        values = []
        weights = [[]]
        f = open(filepath + testgroup + '.kp')
        lines = f.read().splitlines()

        capacities.append(int(lines[2]))
        for line in islice(lines, 4, None):
            data = line.split()
            values.append(int(data[0]))
            weights[0].append(int(data[1]))

        solver = pywrapknapsack_solver.KnapsackSolver(
            pywrapknapsack_solver.KnapsackSolver.
            KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')

        start = time.time()
        solver.Init(values, weights, capacities)
        solver.set_time_limit(300.0)
        computed_value = solver.Solve()
        end = time.time()

        packed_items = []
        packed_weights = []
        total_weight = 0
        string = str(testgroup + '.kp: Total value = ' + str(computed_value))
        a.write(string)
        for i in range(len(values)):
            if solver.BestSolutionContains(i):
                packed_items.append(i)
                packed_weights.append(weights[0][i])
                total_weight += weights[0][i]
        a.write(', Total weight:'+ str(total_weight))
        #a.write(' Packed items:'+ str(packed_items))
        #a.write(' Packed_weights:'+ str(packed_weights))
        a.write(', Time:' + str(end-start))
        a.write('\n\n')
        print('Group: ' + groupname + ' | ' + 'Size: ' + testgroup + ' completed')

    a.close()

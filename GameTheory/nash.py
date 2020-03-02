# %%
import random
import numpy as np

# %%
# Evolve Nash Equilibrium for a simple game:
# (Competitive Coevolutionary approach)
#
# We play a game where we each draw a secret random number uniformly between 0 and 1. 
# We each may re-throw if dissatisfied with our first throw, or me may keep it. 
# We do not know whether or not the other has chosen to re-throw. 
# We then compare our results and he who holds largest number wins $1. 
# What is the best strategy to follow?
#
# O(popSize^2 * generations)
def evolveNash(popSize, generations):
    # Record mean cutoff per generation (is there a better metric?)
    p1Means = []
    p2Means = []

    # Initial cutoff to reroll
    p1Rerolls = [0.5] * popSize
    p2Rerolls = [0.5] * popSize

    for n in range(generations):
        # Evaulate fitness
        p1Fitness = [0] * popSize
        p2Fitness = [0] * popSize
        for i in range(popSize):
            for j in range(popSize):
                # Play the game
                p1Roll = random.random()
                if p1Roll < p1Rerolls[i]: p1Roll = random.random()
                p2Roll = random.random()
                if p2Roll < p2Rerolls[j]: p2Roll = random.random()
                if p1Roll > p2Roll:
                    p1Fitness[i] += 1
                else: 
                    p2Fitness[j] += 1

        # Fitness proportionate selection
        p1FSum = sum(p1Fitness)
        p1FitProps = [f / p1FSum for f in p1Fitness]
        p2FSum = sum(p2Fitness)
        p2FitProps = [f / p2FSum for f in p2Fitness]
        
        p1Rerolls = np.random.choice(p1Rerolls, size=popSize, p=p1FitProps)
        p2Rerolls = np.random.choice(p2Rerolls, size=popSize, p=p2FitProps)

        # Mutate
        norm = np.random.normal(loc=0, scale=0.01, size=popSize)
        p1Rerolls = [ min(p + x, 1) for (p, x) in zip(p1Rerolls, norm)]

        norm = np.random.normal(loc=0, scale=0.01, size=popSize)
        p2Rerolls = [ min(p + x, 1) for (p, x) in zip(p2Rerolls, norm)]

        # Record the means
        p1Means.append(np.mean(p1Rerolls))
        p2Means.append(np.mean(p2Rerolls))

    return p1Means, p2Means
# %%

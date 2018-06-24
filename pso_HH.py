import NeuronSimul
import pygmo as pg
import params






if __name__ == '__main__':
    runner = NeuronSimul.HodgkinHuxley()

    prob = pg.problem(runner)
    print(prob)
    algo = pg.algorithm(pg.bee_colony(gen = 20, limit = 20))
    pop = pg.population(prob, 10)
    pop = algo.evolve(pop)
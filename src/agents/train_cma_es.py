"""Script to train a CMA-ES agent.

Notable env seeds:
- 198804300422
- 850624224

Usage:
    (within src/ directory)
    python -m agents.train_cma_es
"""
from functools import partial

import cma
import fire
import matplotlib.pyplot as plt
import numpy as np
from kaggle_environments import make
from logdir import LogDir
from numpy.core.numeric import Inf

# pylint: disable = import-error
from agents.neural import TOTAL_PARAMS, neural_agent, worker_net
from agents.simple import simple_agent

# pylint: disable = logging-too-many-args
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

MAX_SCORE = -Inf


def simulate(weights, n_evals, seed, gen, debug, logdir: LogDir):
    """Plays the agent with certain weights."""
    global MAX_SCORE

    rng = np.random.default_rng(seed)

    # This way, the first seed is always the seed we pass in.
    env_seeds = [seed] + list(rng.integers(0, 2_147_483_647, n_evals - 1))

    fitnesses = []
    neural_agent_with_weights = partial(neural_agent, weights=weights)

    for s in env_seeds:
        env = make("lux_ai_2021", configuration={"seed": int(s)}, debug=debug)
        steps = env.run([neural_agent_with_weights, simple_agent])

        # Sometimes the reward is None. Bug?

        # Set fitness to be difference between the two player scores.
        fitness = (
            (env.state[0].reward if env.state[0].reward is not None else 0) -
            (env.state[1].reward if env.state[1].reward is not None else 0))

        # Save replays for scores above a certain threshold -- 0 threshold
        # indicates the agent won.
        if fitness >= 0:
            replay = env.toJSON()
            logdir.save_data(
                replay, f"replays/replay-gen_{gen}-seed_{s}-{fitness}.json")

        MAX_SCORE = fitness if fitness > MAX_SCORE else MAX_SCORE

        if debug:
            logger.info("Reward: {}", fitness)
            logger.info("Seed: {}", s)
            logger.info("Max Score: {}", MAX_SCORE)
            logger.info("")

        fitnesses.append(fitness)

    if debug:
        logger.info("Gen {}", gen)
    logger.info("Fitnesses: {}", fitnesses)
    logger.info("Mean: {}", np.mean(fitnesses))
    logger.info("_______________")
    return np.mean(fitnesses)


def train(
        seed: int = 42,  # Master seed.
        sigma: float = 1.0,  # Initial sigma for CMA-ES.
        gens: int = 500,  # Total generations.
        n_evals: int = 5,  # Number of times to eval each solution.
        debug: bool = False,  # Whether to turn on debug mode for envs.
        diagonal: bool = True,  # Whether to use sep-CMA-ES.
):
    """Trains a model with CMA-ES and saves it."""
    logdir = LogDir("cma-es")

    logger.info("TOTAL_PARAMS: {}", TOTAL_PARAMS)
    logger.info("worker_net:\n{}", worker_net)

    # Train with CMA-ES.
    cmaes = cma.CMAEvolutionStrategy(
        np.zeros(TOTAL_PARAMS
                ),  # May consider other initializations. # Pass in npy model
        sigma,
        {
            "seed": seed,
            "CMA_diagonal": diagonal,
        },
    )
    logger.info("Population Size: {}", cmaes.popsize)

    rng = np.random.default_rng(seed)

    env_seeds = [seed] + list(rng.integers(0, 2_147_483_647, n_evals - 1))
    logger.info("Environment Seeds: {}", env_seeds)

    metrics = {
        "best": {
            "x": [],
            "y": [],
        }
    }

    for i in range(1, gens + 1):
        logger.info("===== Generation {} =====", i)
        solutions = cmaes.ask()

        # Use negative here since CMA-ES minimizes and we want to maximize.
        logger.info("Evaluating solutions")
        # Same seed always used in evals, so that scores are comparable.
        objs = [
            -simulate(s, n_evals, seed, i, debug, logdir) for s in solutions
        ]

        cmaes.tell(solutions, objs)
        cmaes.logger.add()
        cmaes.disp()

        metrics["best"]["x"].append(i)
        metrics["best"]["y"].append(-cmaes.result.fbest)

        logger.info("Saving best model")
        np.save(logdir.pfile(f"models/{i}.npy"), -cmaes.result.xbest)

        logger.info("Saving plot of best performance")
        plt.figure()
        plt.plot(metrics["best"]["x"], metrics["best"]["y"])
        plt.title("Best Performance")
        plt.savefig(logdir.file("best.pdf"))
        plt.savefig(logdir.file("best.png"))
        plt.close()

    cmaes.result_pretty()


if __name__ == "__main__":
    fire.Fire(train)

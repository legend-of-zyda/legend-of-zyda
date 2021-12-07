"""Script to train a CMA-ES agent initialized from imitation learning agent.

Usage:
    (within src/ directory)
    python -m agents.train_cma_es_imitation
"""
import multiprocessing

import cma
import fire
import kaggle_environments.envs.lux_ai_2021.lux_ai_2021 as lux_engine
import matplotlib.pyplot as plt
import numpy as np
from cma.fitness_transformations import EvalParallel2
from kaggle_environments import make
from logdir import LogDir
from numpy.core.numeric import Inf

# pylint: disable = import-error
from agents.agent_imitation import (default_imitation_agent_instance,
                                    get_default_model_weights,
                                    imitation_agent_from_weights)

# pylint: disable = logging-too-many-args
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

MAX_WINS = -Inf
episode = 0


def simulate(weights, n_evals, seed, gen, debug, sol_id: int, logdir: LogDir):
    """Plays the agent with certain weights.

    Fitness is counted as how many games the new agent wins against the
    imitation learning agent, so n_evals needs to be high enough.
    """
    global MAX_WINS, episode

    rng = np.random.default_rng(seed)
    # This way, the first seed is always the seed we pass in.
    env_seeds = [seed] + list(rng.integers(0, 2_147_483_647, n_evals - 1))

    new_agent = imitation_agent_from_weights(weights)

    # Simply count wins as the fitness.
    wins = 0
    scores = []

    # Check if new weights are very far away from original.
    #  logger.info("New Weights: {}", new_agent.model.serialize()[:10])
    #  logger.info(
    #      "Original Weights: {}",
    #      np.concatenate([
    #          p.data.cpu().detach().numpy().ravel()
    #          for p in get_default_model().parameters()
    #      ])[:10])

    for s_idx, s in enumerate(env_seeds):

        # TODO - confirm if this doesn't break the algo...
        # skip if it can't beat the best solution so far
        # games_remaining = len(env_seeds) - (s_idx + 1)
        # wins_needed = MAX_WINS - wins + 1

        # if(games_remaining < wins_needed):
        #     continue

        if episode % 1000 == 0:
            # every 1000 episodes, terminate the engine forcefully
            if lux_engine.dimension_process is not None:
                lux_engine.dimension_process.kill()
                lux_engine.dimension_process = None
                episode = 0

        # Alternate sides to prevent overfitting to one side. The best way would
        # be to play each seed on both sides, but that may be too costly.
        side = int(s_idx % 2 == 0)
        env = make("lux_ai_2021", configuration={"seed": int(s)}, debug=True)
        players = ([new_agent, default_imitation_agent_instance] if side == 0
                   else [default_imitation_agent_instance, new_agent])
        steps = env.run(players)
        episode = episode + 1

        # Set score to be difference between the two player scores.
        score = (
            (env.state[0].reward if env.state[0].reward is not None else 0) -
            (env.state[1].reward if env.state[1].reward is not None else 0))
        if side == 1:
            score = -score
        if score > 0:
            wins += 1
        scores.append([side, score])

        logger.info("Seed {} | Won: {}", s_idx, score > 0)
        logger.info("(Side, Score): {} {}", side, score)
        logger.info("    ")

        replay = env.toJSON()
        logdir.save_data(
            replay,
            f"replays/gen_{gen}-sol_{sol_id}-seed_{s}-score_{score}.json",
        )

    # TODO - fix global variable for multiprocessing. each process will have a different MAX_WINS, not really a big deal at the moment
    MAX_WINS = wins if wins > MAX_WINS else MAX_WINS

    logger.info("Wins: {}", wins)
    logger.info("(Side, Score): {}", scores)
    logger.info("Gen {}", gen)
    logger.info("Max Wins So Far {}", MAX_WINS)
    logger.info("_______________")

    # Negate here since function is called differently now
    return -wins


def train(
        seed: int = 42,  # Master seed.
        sigma: float = 0.02,  # Initial sigma for CMA-ES.
        gens: int = 500,  # Total generations.
        checkpoint_freq: int = 1,  # How often to save checkpoints. 
        n_evals: int = 33,  # Number of times to eval each solution.
        debug: bool = False,  # Whether to turn on debug mode for envs.
        diagonal: bool = True,  # Whether to use sep-CMA-ES.
        popsize: int = 10,
        number_of_processes: int = multiprocessing.cpu_count(
        ),  # Number of processes to run in parallel
):
    """Trains a model with CMA-ES and saves it."""
    params = locals()
    logdir = LogDir("cma-es-imitation")
    logdir.save_data(params, "params.json")

    logger.info("Logging Directory: {}", logdir.logdir)

    initial_weights = get_default_model_weights()

    logger.info("TOTAL_PARAMS: {}", len(initial_weights))

    # Train with CMA-ES.
    cmaes = cma.CMAEvolutionStrategy(
        initial_weights,
        sigma,
        {
            "seed": seed,
            "CMA_diagonal": diagonal,
            "popsize": popsize
        },
    )
    logger.info("Population Size: {}", cmaes.popsize)

    rng = np.random.default_rng(seed)

    env_seeds = [seed] + list(rng.integers(0, 2_147_483_647, n_evals - 1))
    logger.info("Environment Seeds: {}", env_seeds)

    metrics = {
        "maxwins": {
            "x": [],
            "y": [],
        }
    }

    with EvalParallel2(fitness_function=simulate,
                       number_of_processes=number_of_processes) as eval_all:
        for i in range(1, gens + 1):
            logger.info("===== Generation {} =====", i)

            logger.info("Evaluating solutions")
            solutions = cmaes.ask()

            # Use negative here since CMA-ES minimizes and we want to maximize. Same seed always used in evals, so that scores are comparable.
            # TODO - refactor to use sol_id again
            sol_id = 1
            cmaes.tell(
                solutions,
                eval_all(solutions,
                         args=(n_evals, seed, i, debug, sol_id, logdir)))
            cmaes.logger.add()
            cmaes.disp()

            metrics["maxwins"]["x"].append(i)
            metrics["maxwins"]["y"].append(-cmaes.result.fbest)

            logger.info("Saving best model")
            np.save(logdir.pfile(f"models/{i}.npy"), cmaes.result.xbest)

            # save cmaes checkpoint file every 50 gens
            if (i % checkpoint_freq == 0):
                checkpoint = cmaes.pickle_dumps()
                open(logdir.pfile(f"cma_checkpoints/{i}.cma_checkpoint"),
                     'wb').write(checkpoint)

            logger.info("Saving plot of best performance")

            plt.figure()
            plt.plot(metrics["maxwins"]["x"], metrics["maxwins"]["y"])
            plt.title("Max Wins Attained So Far")
            plt.savefig(logdir.file("maxwins.pdf"))
            plt.savefig(logdir.file("maxwins.png"))
            plt.close()

            cmaes.result_pretty()


if __name__ == "__main__":
    fire.Fire(train)

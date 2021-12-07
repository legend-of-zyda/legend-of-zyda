"""Script to train a CMA-ES agent initialized from imitation learning agent.

Usage:
    (within src/ directory)
    python -m agents.train_cma_es_imitation

    # For debugging.
    python -m agents.train_cma_es_imitation \
        --n-evals 1 --workers 1 --gens 1 --debug
"""
from cmaes import CMA
import fire
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from dask.distributed import Client, LocalCluster
from kaggle_environments import make
from logdir import LogDir

# pylint: disable = import-error
from agents.agent_imitation import (get_default_model_weights,
                                    imitation_agent_from_weights)

# pylint: disable = logging-too-many-args
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def simulate(weights, head_only, n_evals, seed, gen, debug, sol_id: int,
             logdir: LogDir):
    """Plays the agent with certain weights.

    Fitness is counted as how many games the new agent wins against the
    imitation learning agent, so n_evals needs to be high enough.
    """
    # Necessary to import this here because Dask cannot serialize a function
    # which captures a JIT module as a global variable.
    # pylint: disable = import-outside-toplevel
    from agents.agent_imitation import (
        DEVICE, default_imitation_agent_instance,
        default_imitation_agent_instance_head_only)
    import kaggle_environments.envs.lux_ai_2021.lux_ai_2021 as lux_engine

    # Terminate lux engine to prevent JSONDecodeError / memory leaks.
    if lux_engine.dimension_process is not None:
        logger.info("Killing lux engine")
        lux_engine.dimension_process.kill()
        lux_engine.dimension_process = None

    opponent = (default_imitation_agent_instance_head_only
                if head_only else default_imitation_agent_instance)
    logger.info("DEVICE: {}", DEVICE)
    torch.set_num_threads(1)

    rng = np.random.default_rng(seed)

    # This way, the first seed is always the seed we pass in.
    env_seeds = [seed] + list(rng.integers(0, 2_147_483_647, n_evals - 1))

    new_agent = imitation_agent_from_weights(weights, head_only)

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
        # Alternate sides to prevent overfitting to one side. The best way would
        # be to play each seed on both sides, but that may be too costly.
        side = 1 - (s_idx % 2)

        env = make("lux_ai_2021", configuration={"seed": int(s)}, debug=debug)
        players = [new_agent, opponent] if side == 0 else [opponent, new_agent]
        steps = env.run(players)  # pylint: disable = unused-variable

        # Sometimes the reward is None. Bug?

        # Set score to be difference between the two player scores.
        score = (
            (env.state[0].reward if env.state[0].reward is not None else 0) -
            (env.state[1].reward if env.state[1].reward is not None else 0))
        if side == 1:
            score = -score
        if score > 0:
            wins += 1
        scores.append([side, score])

        if debug:
            replay = env.toJSON()
            logdir.save_data(
                replay,
                f"replays/gen_{gen}-sol_{sol_id}-seed_{s}-score_{score}.json",
            )

    logger.info("Generation {}", gen)
    logger.info("Solution {} | Wins: {}", sol_id, wins)
    logger.info("(Side, Score): {}", scores)
    logger.info("_______________")

    # Add a small amount here such that solutions in current gens are favored.
    return wins + gen * 1e-6


def train(
        seed: int = 117,  # Master seed.
        sigma: float = 0.02,  # Initial sigma for CMA-ES.
        gens: int = 10000,  # Total generations.
        n_evals: int = 30,  # Number of times to eval each solution.
        debug: bool = True,  # Whether to turn on debug mode for envs.
        diagonal: bool = False,  # Use sep-CMA-ES. Only use if head_only=False.
        workers: int = 6,  # Workers.
        head_only: bool = True,  # Whether to train only the head.
        popsize: int = 6,
):
    """Trains a model with CMA-ES and saves it."""
    params = locals()
    logdir = LogDir("cma-es-imitation")
    logdir.save_data(params, "params.json")

    logger.info("Logging Directory: {}", logdir.logdir)

    # Set up Dask.
    client = Client(
        LocalCluster(n_workers=workers, threads_per_worker=1, processes=True))
    logger.info("Dask Cluster config: {}", client.ncores())

    initial_weights = get_default_model_weights(head_only)
    logger.info("TOTAL_PARAMS: {}", len(initial_weights))

    rng = np.random.default_rng(seed)

    env_seeds = [seed] + list(rng.integers(0, 2_147_483_647, n_evals - 1))
    logger.info("Environment Seeds: {}", env_seeds)

    # Train with CMA-ES.
    cmaes = CMA(mean=initial_weights, sigma=sigma, seed=seed, population_size=popsize)
    logger.info("Population Size: {}", cmaes.population_size)

    n_restarts = 0  # A small restart doesn't count in the n_restarts
    small_n_eval, large_n_eval = 0, 0
    popsize0 = popsize
    inc_popsize = 2

    # Initial run is with "normal" population size; it is
    # the large population before first doubling, but its
    # budget accounting is the same as in case of small
    # population.
    poptype = "small"

    best_sol = []
    best_fitness = 0

    metrics = {
        "best": {
            "x": [],
            "y": [],
        },
        "mean": {
            "x": [],
            "y": [],
        },
    }

    for generation in range(200):
        results = []
        solutions = []
        futures = []
        for sol_id in range(cmaes.population_size):
            sol = cmaes.ask()
            # futures.append(
            future = client.submit(
                simulate,
                sol,
                head_only,
                n_evals,
                seed,
                generation,
                debug,
                sol_id,
                logdir,
                pure=False,
            )
            futures.append(future)
            solutions.append(sol)
            # )
            # print(f"#{generation} {value} (x1={sol[0]}, x2 = {sol[1]})")

        # Use negative here since CMA-ES minimizes and we want to maximize.
        objs = -np.array(client.gather(futures))
        for fit_id, fitness in enumerate(objs):
            if(-fitness > best_fitness):
                best_fitness = -fitness
                best_sol = solutions[fit_id]
            results.append((solutions[fit_id], fitness))

        cmaes.tell(results)

        logger.info("Saving best model")
        np.save(logdir.pfile(f"models/{generation}.npy"), best_sol)

        metrics["best"]["x"].append(generation)
        metrics["best"]["y"].append(best_fitness)
        metrics["mean"]["x"].append(generation)
        metrics["mean"]["y"].append(-np.mean(objs))

        logger.info("Best performance over gens: {}", metrics["best"]["y"])
        logger.info("Mean performance over gens: {}", metrics["mean"]["y"])

        logger.info("Saving plot of best performance")
        plt.figure()
        plt.plot(metrics["best"]["x"], metrics["best"]["y"])
        plt.title("Best Performance")
        plt.savefig(logdir.file("best.pdf"))
        plt.savefig(logdir.file("best.png"))
        plt.close()

        logger.info("Saving plot of mean performance")
        plt.figure()
        plt.plot(metrics["mean"]["x"], metrics["mean"]["y"])
        plt.title("Mean Performance")
        plt.savefig(logdir.file("mean.pdf"))
        plt.savefig(logdir.file("mean.png"))
        plt.close()

        if cmaes.should_stop():
            n_eval = cmaes.population_size * cmaes.generation
            if poptype == "small":
                small_n_eval += n_eval
            else:  # poptype == "large"
                large_n_eval += n_eval

            if small_n_eval < large_n_eval:
                poptype = "small"
                popsize_multiplier = inc_popsize ** n_restarts
                popsize = round(math.floor(
                    popsize0 * popsize_multiplier ** (np.random.uniform() ** 2)
                ))
            else:
                poptype = "large"
                n_restarts += 1
                popsize = popsize0 * (inc_popsize ** n_restarts)

            # mean = lower_bounds + (np.random.rand(2) * (upper_bounds - lower_bounds))
            cmaes = CMA(
                mean=initial_weights,
                sigma=sigma,
                # bounds=bounds,
                population_size=popsize,
            )
            print("Restart CMA-ES with popsize={} ({})".format(popsize, poptype))

if __name__ == "__main__":
    fire.Fire(train)

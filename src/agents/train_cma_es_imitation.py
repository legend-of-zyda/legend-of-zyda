"""Script to train a CMA-ES agent initialized from imitation learning agent.

Usage:
    (within src/ directory)
    python -m agents.train_cma_es_imitation

    # For debugging.
    python -m agents.train_cma_es_imitation \
        --n-evals 1 --workers 1 --gens 1 --debug
"""
import cma
import fire
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

    logger.info("Solution {} | Wins: {}", sol_id, wins)
    logger.info("(Side, Score): {}", scores)
    logger.info("_______________")

    # Add a small amount here such that solutions in current gens are favored.
    return wins + gen * 1e-6


def train(
        seed: int = 42,  # Master seed.
        sigma: float = 0.02,  # Initial sigma for CMA-ES.
        gens: int = 10000,  # Total generations.
        n_evals: int = 1,  # Number of times to eval each solution.
        debug: bool = False,  # Whether to turn on debug mode for envs.
        diagonal: bool = False,  # Use sep-CMA-ES. Only use if head_only=False.
        workers: int = 1,  # Workers.
        head_only: bool = True,  # Whether to train only the head.
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

    # Train with CMA-ES.
    cmaes = cma.CMAEvolutionStrategy(
        initial_weights,
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
        },
        "mean": {
            "x": [],
            "y": [],
        },
    }

    for i in range(1, gens + 1):
        logger.info("===== Generation {} =====", i)
        solutions = cmaes.ask()

        logger.info("Evaluating solutions")

        # Same seed always used in evals, so that scores are comparable.
        futures = [
            client.submit(
                simulate,
                sol,
                head_only,
                n_evals,
                seed,
                i,
                debug,
                sol_id,
                logdir,
                pure=False,
            ) for sol_id, sol in enumerate(solutions)
        ]

        # Use negative here since CMA-ES minimizes and we want to maximize.
        objs = -np.array(client.gather(futures))

        cmaes.tell(solutions, objs)
        cmaes.logger.add()
        cmaes.disp()

        metrics["best"]["x"].append(i)
        metrics["best"]["y"].append(-cmaes.result.fbest)
        metrics["mean"]["x"].append(i)
        metrics["mean"]["y"].append(-np.mean(objs))

        logger.info("Best performance over gens: {}", metrics["best"]["y"])
        logger.info("Mean performance over gens: {}", metrics["mean"]["y"])

        logger.info("Saving best model")
        np.save(logdir.pfile(f"models/{i}.npy"), cmaes.result.xbest)

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

    cmaes.result_pretty()


if __name__ == "__main__":
    fire.Fire(train)

"""Runs the improved imitation agent vs original imitation agent.

Usage:
    python -m agents.test_imitation
"""
import json

import fire
from kaggle_environments import make

from agents.agent_imitation import (default_imitation_agent_instance,
                                    improved_imitation_agent_instance)


def main(seed: int = 42):
    env = make("lux_ai_2021", configuration={"seed": seed}, debug=True)
    env.run([
        improved_imitation_agent_instance,
        default_imitation_agent_instance,
    ])

    # Set score to be difference between the two player scores.
    score = ((env.state[0].reward if env.state[0].reward is not None else 0) -
             (env.state[1].reward if env.state[1].reward is not None else 0))
    print("Score:", score)

    with open("imitation_replay.json", "w") as file:
        json.dump(env.toJSON(), file)
        print("Replay saved to imitation_replay.json")


if __name__ == "__main__":
    fire.Fire(main)

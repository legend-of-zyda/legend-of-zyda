"""Chooses an agent based on current agent_id.txt."""
from pathlib import Path

# pylint: disable = import-error
from agents.CARB import CARB_agent
from agents.neural import neural_agent
from agents.simple import simple_agent
from agents.agent_imitation import agent_imitation

# This line is beautiful.
agent_id = (Path(__file__).parent / "agent_id.txt").open().read().strip()

# To add a new agent:
# 1. Write your agent in the agents/ dir
# 2. Import the agent above.
# 3. Choose an agent_id and add a mapping from id to agent below.
agent = {
    "simple": simple_agent,
    "CARB": CARB_agent,
    "neural": neural_agent,
    "Imitation": agent_imitation
}[agent_id]

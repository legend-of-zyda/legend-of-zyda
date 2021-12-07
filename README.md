# Legend of Zyda

Agents for Lux AI challenge. Part of the project "Reinforcement Learning for the
Lux AI Challenge," a project for USC's CSCI 527. For more info, please visit our
website: <https://legend-of-zyda.netlify.app>

## Contributors

- [Will Borie](https://www.willborie.com/)
- [Jordan Ishii](https://www.linkedin.com/in/jordan-ishii-a13b67104/)
- [Felix Loesing](https://www.linkedin.com/in/felix-loesing/)
- [Bryon Tjanaka](https://btjanaka.net/)
- [Robert Trybula](https://www.linkedin.com/in/rob-trybula/)

## Instructions

### Environment

Create the environment in the `./env` directory.

```bash
conda env create -f environment.yml --prefix ./env
```

And activate with

```bash
conda activate ./env
```

### Writing an Agent

1. Write your agent in `src/agents/`.
1. Import your agent in `src/agent.py`.
1. Choose an agent id and map from agent id to the agent function in
   `src/agent.py`.
1. Switch the agent id by modifying `src/agent_id.txt` or using the
   `change_id.sh` script.

### Scripts

Scripts are located under the `scripts/` directory.

| Script                | Description                         |
| --------------------- | ----------------------------------- |
| `selfplay.sh`         | Plays an agent against itself.      |
| `build_submission.sh` | Prepares a tar file for submission. |
| `change_id.sh`        | Change agent id.                    |

### Connecting to a Remote Jupyter Notebook

1. Install jupyter lab.
   ```bash
   pip install jupyterlab
   ```
1. Start jupyter lab with all IPs available.
   ```bash
   jupyter lab --ip 0.0.0.0
   ```
1. Start an SSH tunnel from your machine to the remote server.
   ```bash
   ssh -p SSH_PORT -N -L 8888:HOSTNAME:8888 USERNAME@HOSTNAME
   ```
1. Open the localhost link that was output by jupyter lab, e.g.
   http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXX

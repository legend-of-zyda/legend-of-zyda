# Runs the agent against itself.
#
# Usage:
#   conda activate ./env
#   scripts/selfplay.sh DIRECTORY SEED

directory="$1"
seed="$2"

# Absolute path to lux-ai-2021 command.
# If on MacOS, install realpath with "brew install coreutils"
cmd="$(realpath $(which lux-ai-2021 | head -n 1))"

which python3
cd $directory
$cmd main.py main.py --python=python3 --out=replay.json --seed=$seed

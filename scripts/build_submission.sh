# Builds submission.
#
# Make sure to change the agent id (see change_id.sh) if needed.
#
# Usage:
#   scripts/build_submission.sh DIRECTORY

directory="$1"
filename="submission.tar.gz"
tmpdir="$(mktemp -d)"

cd "$directory"

# Clean directory.
find . -name __pycache__ -type d -exec rm -rf {} \;
mv logs results outcmaes errorlogs replay.json replays "$tmpdir"

# Clear old submission.
rm -vf "$filename"

# Make new submission.
tar -czvf "$filename" *

# Bring back files.
mv $tmpdir/* .

echo "===== Submission created in: ====="
echo "${directory}/${filename}"

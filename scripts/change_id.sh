# Changes the current agent id.
#
# Usage:
#   scripts/change_id.sh AGENT_ID

agent_id=$1
echo $1 > src/agent_id.txt

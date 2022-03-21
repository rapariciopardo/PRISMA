numNodes=10
basePort=6655
for ((i=0; i<=numNodes; i++)); do
	agentPort=$((basePort+i))
	fuser -k $((basePort+i))/tcp
done
numNodes=23
basePort=6555
for ((i=0; i<=numNodes; i++)); do
	agentPort=$((basePort+i))
	fuser -k $((basePort+i))/tcp
done

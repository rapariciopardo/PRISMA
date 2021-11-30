#!/bin/bash
numNodes=26
basePort=5555
for ((i=0; i<=numNodes; i++)); do
	agentPort=$((basePort+i))
	fuser -k $((basePort+i))/tcp
done
echo All done
#!/bin/bash
numNodes=5
basePort=5555
for (( i=0; i<=numNodes; i++ ))
do
	agentPort=$((basePort+i))
	output=outputs/out_"$agentPort".txt
	nohup python3 test.py --start=0 --index=$((i)) --port=$((basePort+i))  > "$output" &
	echo $agentPort
done
#echo All done

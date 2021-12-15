from graph import *

g = Graph(5, 0)
g.openFile()
g.dijkstra()
g.getRoutingTable()
print(g.D)
print(g.Parent)
print(g.RoutingTable)
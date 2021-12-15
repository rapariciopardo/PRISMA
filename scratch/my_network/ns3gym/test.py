from graph import *

g = Graph(5)
g.openFile()
g.dijkstra(3)
print(g.D)
print(type(g.D))
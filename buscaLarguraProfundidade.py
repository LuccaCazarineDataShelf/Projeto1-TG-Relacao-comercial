from GrafoProjeto import GrafoMatriz

g = GrafoMatriz(rotulado=False)

for vertice in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
    g.adicionarVertice(vertice)

g.insereA('a', 'b')
g.insereA('a', 'c')
g.insereA('a', 'e')

g.insereA('b', 'd')
g.insereA('b', 'e')

g.insereA('c', 'f')
g.insereA('c', 'g')


g.insereA('d', 'h')

g.insereA('e', 'h')

g.insereA('f', 'e')
g.insereA('f', 'g')

g.insereA('g', 'h')

g2 = GrafoMatriz(rotulado=False)

for vertice in ['a', 'b', 'c', 'd']:
    g2.adicionarVertice(vertice)


g2.insereA('a', 'b')
g2.insereA('a', 'c')

g2.insereA('b', 'a')
g2.insereA('b', 'd')

g2.insereA('c', 'a')
g2.insereA('c', 'd')

g2.insereA('d', 'b')
g2.insereA('d', 'c')


print("\nGRAFO 1:")
g.show()
print("\nGRAFO 2:")
g2.show()

print("\nBUSCA EM PROFUNDIDADE:")
print("\nGRAFO 1:")
print(g.percursoProfundidade('a'))

print("\nGRAFO 2:")
print(g2.percursoProfundidade('a'))

print("\nBUSCA EM LARGURA:")
print("\nGRAFO 1:")
print(g.percursoLargura('a'))

print("\nGRAFO 2:")
print(g2.percursoLargura('a'))

from grafoLista import TGrafo, GrafoLista
from grafoMatriz import GrafoMatriz, GrafoMatrizND

print('Lista 1 Teoria dos Grafos')
# Coluna Recebe e Linha envia Linha -> coluna
print('\nExercicio 1')
ex1 = GrafoMatriz(5)

ex1.insereA(0, 1)
ex1.insereA(2, 1)
ex1.insereA(3, 1)
ex1.insereA(4, 1)

print("Grau de entrada do vértice não rotulado 1:", ex1.inDegree(1))    #expected 4
print("Grau de entrada do vértice não rotulado 0:", ex1.inDegree(0))    #expected 0

print('\nExercicio 2')
ex2 = GrafoMatriz(5)
ex2rotulado = GrafoMatriz(5, True)

ex2.insereA(0, 1)
ex2.insereA(0, 2)
ex2.insereA(0, 3)
ex2.insereA(0, 4)

ex2rotulado.insereA(0, 1, 4.0)
ex2rotulado.insereA(0, 2, 8.0)
ex2rotulado.insereA(1, 3, 16.0)
ex2rotulado.insereA(1, 4, 32.0)

print("Grau de saída do vértice 0:", ex2.outDegree(0))   #expected 4
print("Grau de saída do vértice 1:", ex2.outDegree(1))   #expected 0

print("Grau de saída do vértice rotulado 0:", ex2rotulado.outDegree(0))   #expected 2
print("Grau de saída do vértice rotulado 1:", ex2rotulado.outDegree(1))   #expected 2

print('\nExercicio 3')

ex3 = GrafoMatriz(5)
ex3rotulado = GrafoMatriz(5, True)

ex3.insereA(0, 1)
ex3.insereA(0, 2)
ex3.insereA(0, 3)
ex3.insereA(0, 4)
ex3.insereA(1, 0)
ex3.insereA(2, 0)
ex3.insereA(3, 0)
ex3.insereA(4, 0)

ex3rotulado.insereA(0, 1, 4.0)
ex3rotulado.insereA(0, 2, 8.0)
ex3rotulado.insereA(2, 3, 16.0)
ex3rotulado.insereA(1, 4, 32.0)



print("Grau do vértice 0:", ex3.degree(0))   #expected 8
print("Grau do vértice 1:", ex3.degree(1))   #expected 2

print("Grau do vértice rotulado 0:", ex3rotulado.degree(0))       #expected 2
print("Grau do vértice rotulado 2:", ex3rotulado.degree(2))       #expected 2

print('\nExercicio 4')

ex4 = GrafoMatriz(5)
ex4rotulado = GrafoMatriz(5, True)

ex4.insereA(0, 1)
ex4.insereA(0, 2)
ex4.insereA(0, 3)
ex4.insereA(0, 4)
ex4.insereA(2, 1)
ex4.insereA(3, 1)

ex4rotulado.insereA(0, 1, 4.0)
ex4rotulado.insereA(0, 2, 8.0)
ex4rotulado.insereA(1, 3, 16.0)
ex4rotulado.insereA(1, 4, 32.0)

print("Vértice 0 é fonte?:", ex4.isSource(0))   #expected 1
print("Vértice 1 é fonte?:", ex4.isSource(1))   #expected 0

print("Vértice rotulado 0 é fonte?:", ex4rotulado.isSource(0)) #expected 1
print("Vértice rotulado 1 é fonte?:", ex4rotulado.isSource(1)) #expected 0

print('\nExercicio 5')

ex5 = GrafoMatriz(5)
ex5rotulado = GrafoMatriz(5, True)

ex5.insereA(0, 1)
ex5.insereA(0, 2)
ex5.insereA(0, 3)
ex5.insereA(0, 4)
ex5.insereA(2, 1)
ex5.insereA(3, 1)

ex5rotulado.insereA(0, 1, 4.0)
ex5rotulado.insereA(0, 2, 8.0)
ex5rotulado.insereA(1, 3, 16.0)
ex5rotulado.insereA(1, 4, 32.0)

print("Vértice 0 é Sorvedouro?:", ex5.isSorvedouro(0)) #expected 0
print("Vértice 1 é Sorvedouro?:", ex5.isSorvedouro(1)) #expected 1

print("Vértice rotulado 0 é Sorvedouro?:", ex5rotulado.isSorvedouro(0)) #expected 0
print("Vértice rotulado 3 é Sorvedouro?:", ex5rotulado.isSorvedouro(3)) #expected 1

print('\nExercicio 6')

ex6 = GrafoMatriz(4)
ex6rotulado = GrafoMatriz(4, True)

ex6.insereA(0, 1, 1)
ex6.insereA(1, 0, 1)
ex6.insereA(2, 3, 2)
ex6.insereA(3, 2, 2)
ex6.show()
print('\nO grafo é simetrico?', ex6.simetrico()) #expected True

ex6.insereA(1, 2, 4)
ex6.show()
print('\nO grafo é simetrico?', ex6.simetrico()) #expected False

ex6rotulado.insereA(0, 1, 1)
ex6rotulado.insereA(1, 0, 1)
ex6rotulado.insereA(2, 3, 2)
ex6rotulado.insereA(3, 2, 2)
ex6rotulado.show()
print('\nO grafo rotulado é simetrico?', ex6rotulado.simetrico()) #expected True

ex6rotulado.insereA(1, 2, 4)
ex6rotulado.show()
print('\nO grafo rotulado é simetrico?', ex6rotulado.simetrico()) #expected False

print('\nExercicio 7')

ex7 = GrafoMatriz()

ex7.lerArquivo('grafo.txt')
ex7.show()

print('\nExercicio 8')

ex8 = GrafoMatrizND(4)
ex8rotulado = GrafoMatrizND(4, True)

ex8.show()
print('\nInserindo arestas no grafo')
ex8.insereA(0,1)
ex8.insereA(2,1)
ex8.insereA(3,2)
ex8.show()

print('\nRemovendo aresta no grafo')
ex8.removeA(1,2)
ex8.show()


ex8rotulado.show()
print('\nInserindo arestas no grafo')
ex8rotulado.insereA(0,1)
ex8rotulado.insereA(2,1)
ex8rotulado.insereA(3,2)
ex8rotulado.show()

print('\nRemovendo aresta no grafo')
ex8rotulado.removeA(1,2)
ex8rotulado.show()

print('\nExercico 9')

ex9 = GrafoMatrizND(5)

ex9.insereA(0,1)
ex9.insereA(0,4)
ex9.insereA(1,2)
ex9.insereA(3,1)
ex9.insereA(4,1)
ex9.insereA(4,3)
ex9.show()
print('\nGrau do vertice 0 é:', ex9.grau(0)) #expected 2
print('\nGrau do vertice 1 é:', ex9.grau(1)) #expected 4
print('\nGrau do vertice 2 é:', ex9.grau(2)) #expected 1
print('\nGrau do vertice 3 é:', ex9.grau(3)) #expected 2
print('\nGrau do vertice 4 é:', ex9.grau(4)) #expected 3

print('\nExercicio 11')

ex11 = GrafoMatriz(5)
ex11rotulado = GrafoMatriz(5, True)

ex11.insereA(0, 1)
ex11.insereA(0, 2)
ex11.insereA(1, 3)
ex11.insereA(2, 4)

print('\nANTES')
ex11.show()
ex11.removeV(2)
print('\nDEPOIS')

ex11.show()

ex11rotulado.insereA(0, 1, 4.0)
ex11rotulado.insereA(0, 2, 8.0)
ex11rotulado.insereA(2, 3, 16.0)
ex11rotulado.insereA(1, 4, 32.0)

print('\nANTES')
ex11rotulado.show()
ex11rotulado.removeV(1)
print('\nDEPOIS')

ex11rotulado.show()

print('\nExercicio 12')

ex12 = GrafoMatrizND (4)
ex12.insereA(0, 1)
ex12.insereA(0, 2)
ex12.insereA(0, 3)
ex12.insereA(1, 2)
ex12.insereA(1, 3)
ex12.show()
print('\nEste grafo está completo?', ex12.completo())

print('\nAdcionando a aresta que falta')
ex12.insereA(2, 3)
ex12.show()
print('\nEste grafo está completo?', ex12.completo())

print('\nExercicio 13')

ex13 = GrafoMatriz(3)
ex13rotulado = GrafoMatriz(3, True)

ex13.insereA(0, 1)
ex13.insereA(0, 2)
ex13.insereA(1, 0)
ex13.insereA(1, 2)
ex13.insereA(2, 0)
ex13.show()
print('\nO grafo não rotulado é completo? : ', ex13.isComplete()) #expected False

ex13rotulado.insereA(0, 1, 4.0)
ex13rotulado.insereA(0, 2, 8.0)
ex13rotulado.insereA(1, 0, 16.0)
ex13rotulado.insereA(1, 2, 32.0)
ex13rotulado.insereA(2, 0, 64.0)
ex13rotulado.insereA(2, 1, 128.0)
ex13rotulado.show()
print('\nO grafo rotulado é completo? : ', ex13rotulado.isComplete()) #expected True

print('\nExercicio 14')

ex14 = GrafoMatriz(3)
ex14rotulado = GrafoMatriz(3, True)

ex14.insereA(0, 1)
ex14.insereA(0, 2)
ex14.insereA(1, 0)
ex14.insereA(2, 0)

print('\nGrafo original: ')
ex14.show()
print('\nGrafo complementar: ')

ex14.complementar().show()

ex14rotulado.insereA(0, 1, 4.0)
ex14rotulado.insereA(0, 2, 8.0)
ex14rotulado.insereA(1, 2, 32.0)
ex14rotulado.insereA(2, 1, 128.0)
print('\nGrafo rotulado original: ')
ex14rotulado.show()
print('\nGrafo rotualdo complementar: ')
ex14rotulado.complementar().show()

print('\nNão dirigido')

ex14 = GrafoMatrizND(4)

ex14.insereA(0, 1)
ex14.insereA(0, 2)
ex14.insereA(1, 3)

print("Matriz de adjacência antes de calcular o complementar:")
ex14.show()

ex14_complementar = ex14.complementar()

print("Matriz de adjacência do grafo complementar nao dirigido:")
ex14_complementar.show()

print('\nExercicio 15')

ex15 = GrafoMatriz(5)
ex15rotulado = GrafoMatriz(5, True)

ex15.insereA(0, 1)
ex15.insereA(1, 2)
ex15.insereA(2, 3)
ex15.insereA(3, 4)

ex15rotulado.insereA(0, 1, 4.0)
ex15rotulado.insereA(1, 2, 8.0)
ex15rotulado.insereA(2, 3, 16.0)
ex15rotulado.insereA(3, 4, 32.0)

print("Grafo conectado:", ex15.isConected())  # expected 0 (grafo conectado)
print("Grafo conectado (rotulado):", ex15rotulado.isConected())  # expected 0 (grafo conectado)

ex15 = GrafoMatriz(5)
ex15rotulado = GrafoMatriz(4, True)

ex15.insereA(0, 1)
ex15.insereA(2, 3)

ex15rotulado.insereA(0, 1, 4.0)
ex15rotulado.insereA(2, 3, 8.0)

print("Grafo desconectado:", ex15.isConected())  # expected 1 (grafo não conectado)
print("Grafo desconectado (rotulado):", ex15rotulado.isConected())  # expected 1 (grafo não conectado)

print('\nExercico 16')

ex16 = GrafoMatriz(4)
ex16.insereA(0, 1)
ex16.insereA(1, 2)
ex16.insereA(2, 3)
ex16.insereA(3, 0)
ex16.insereA(2, 1)
ex16.insereA(3, 2)
ex16.insereA(1, 0)
ex16.insereA(0, 3)

ex16.show()
print("")
print("Categoria de Conexidade: ", ex16.categoriaConexidade(), "\n")

ex16.removeA(3, 2)
ex16.show()
print("\nCategoria de Conexidade: ", ex16.categoriaConexidade(), "\n")

ex16.removeA(0, 2)
ex16.removeA(0, 3)
ex16.removeA(1, 0)
ex16.removeA(2, 0)
ex16.removeA(2, 1)
ex16.removeA(3, 0)
ex16.removeA(3, 1)
ex16.removeA(3, 2)


ex16.show()
print("\nCategoria de Conexidade: ", ex16.categoriaConexidade(), "\n")

ex16.removeA(0, 1)
ex16.removeA(1, 0)
ex16.removeA(3, 0)
ex16.removeA(2, 0)
ex16.removeA(0, 2)
ex16.removeA(0, 3)

ex16.show()
print("")
print("Categoria de Conexidade: ", ex16.categoriaConexidade(), "\n")

print('\nExercico 17')

ex17 = GrafoMatriz(5)

ex17.insereA(0, 1)
ex17.insereA(1, 2)
ex17.insereA(3, 4)
ex17.insereA(1, 4)
ex17.insereA(2, 4)

print("Matriz não reduzida")
ex17.show()

print("\nMatriz reduzida")
matriz_reduzida = ex17.grafoReduzido()
for linha in matriz_reduzida:
    print(linha)


print('\nExercicio 19')

ex19 = GrafoLista(4)
ex19.insereA(0, 1)
ex19.insereA(0, 2)
ex19.insereA(2, 1)
ex19.insereA(2, 3)
ex19.insereA(1, 3)
print("O grau de entrada do vertice v é: " , ex19.inDegree(2), "\n")

print('\nExercicio 20')

ex20 = GrafoLista(4)
ex20.insereA(0, 1)
ex20.insereA(0, 2)
ex20.insereA(2, 1)
ex20.insereA(2, 3)
ex20.insereA(1, 3)
print("O grau de saída do vertice v é: " , ex20.outDegree(2), "\n")

print('\nExercicio 21')

ex21 = GrafoLista(4)
ex21.insereA(0, 1)
ex21.insereA(0, 2)
ex21.insereA(2, 1)
ex21.insereA(2, 3)
ex21.insereA(1, 3)
print("O grau do vertice v é: " , ex21.degree(2), "\n")

print('\nExercicio 22')

ex22 = GrafoLista(4)
ex22.insereA(0, 1)
ex22.insereA(0, 2)
ex22.insereA(2, 1)
ex22.insereA(2, 3)
ex22.insereA(1, 3)

grafoAux = GrafoLista(4)
grafoAux.insereA(0, 1)
grafoAux.insereA(0, 2)
grafoAux.insereA(2, 1)
grafoAux.insereA(2, 3)
grafoAux.insereA(1, 3)

if ex22.isEqual(ex22,grafoAux):
    print("\n\nOs grafos são iguais.")
else:
    print("Os grafos são diferentes.")

ex22.removeA(0,1)

if ex22.isEqual(ex22,grafoAux):
    print("\n\nOs grafos são iguais.")
else:
    print("Os grafos são diferentes.")


print('\nExercicio 23')

ex23 = GrafoLista(4)
ex23.insereA(0, 1)
ex23.insereA(0, 2)
ex23.insereA(2, 1)
ex23.insereA(2, 3)
ex23.insereA(1, 3)

matriz = ex23.listToMatrix()
print("\nMatriz de adjacência:")
for linha in matriz:
    print(linha)

lista_convertida = ex23.matrixToList(matriz)

print('\nExercicio 24')

ex24 = TGrafo(5)

ex24.insereA(0, 1)
ex24.insereA(0, 2)
ex24.insereA(1, 3)
ex24.insereA(1, 4)

print("Lista de adjacência antes da inversão:")
ex24.show()
ex24.inverter_adj()

print("Lista de adjacência após a inversão:")
ex24.show()

print('\nExercicio 25')

ex25 = TGrafo(5)

ex25.insereA(0, 1)
ex25.insereA(0, 2)
ex25.insereA(1, 3)
ex25.insereA(1, 4)

print("Lista de adjacência:")
ex25.show()

print("Vértice 0 é uma fonte?", ex25.isSource(0))
print("Vértice 1 é uma fonte?", ex25.isSource(1))
print("Vértice 3 é uma fonte?", ex25.isSource(3))

print('\nExercicio 26')

ex26 = TGrafo(5)

ex26.insereA(0, 1)
ex26.insereA(0, 2)
ex26.insereA(1, 3)
ex26.insereA(1, 4)

print("Lista de adjacência:")
ex26.show()

print("Vértice 0 é um sorvedouro?", ex26.isSorvedouro(0))
print("Vértice 1 é um sorvedouro?", ex26.isSorvedouro(1))
print("Vértice 3 é um sorvedouro?", ex26.isSorvedouro(3))
print("Vértice 4 é um sorvedouro?", ex26.isSorvedouro(4))


print('\nExercicio 27')

ex27 = TGrafo(5)

ex27.insereA(0, 1)
ex27.insereA(1, 0)
ex27.insereA(1, 2)
ex27.insereA(2, 1)
ex27.insereA(3, 4)
ex27.insereA(4, 3)

print("Lista de adjacência:")
ex27.show()

print("O grafo é simétrico?", ex27.isSimetric())

ex27 = TGrafo(5)

ex27.insereA(0, 1)
ex27.insereA(1, 2)
ex27.insereA(3, 4)

print("Lista de adjacência:")
ex27.show()

print("O grafo é simétrico?", ex27.isSimetric())

print('\nExercicio 28')


ex28 = TGrafo.lerArquivo('grafo.txt')
ex28.show()

print('\nExercicio 29')

ex29 = TGrafo(5)

ex29.insereA(0, 1)
ex29.insereA(0, 2)
ex29.insereA(1, 3)
ex29.insereA(1, 4)
ex29.insereA(2, 3)

print("Lista de adjacência antes da remoção do vértice:")
ex29.show()

ex29.remover_vertice(1)

print("Lista de adjacência após a remoção do vértice 1:")
ex29.show()

print('\nExercicio 30')

ex30 = GrafoLista(5)

ex30.insereA(0, 1)
ex30.insereA(0, 2)
ex30.insereA(1, 3)
ex30.insereA(1, 4)
ex30.insereA(2, 3)

print("Lista de adjacência antes da remoção do vértice:")
ex30.show()

ex30.remover_vertice(1)

print("Lista de adjacência após a remoção do vértice 1:")
ex30.show()


print('\nExercicio 31')

ex31d = GrafoLista(5)
ex31d.insereA(0, 1)
ex31d.insereA(1, 2)
ex31d.insereA(2, 3)
ex31d.show()

print("O grafo é completo?", ex31d.isComplete())

ex31d.insereA(0, 2)
ex31d.insereA(0, 3)
ex31d.insereA(0, 4)
ex31d.insereA(1, 0)
ex31d.insereA(1, 3)
ex31d.insereA(1, 4)
ex31d.insereA(2, 0)
ex31d.insereA(2, 1)
ex31d.insereA(2, 4)
ex31d.insereA(3, 0)
ex31d.insereA(3, 1)
ex31d.insereA(3, 2)
ex31d.insereA(3, 4)
ex31d.insereA(4, 0)
ex31d.insereA(4, 1)
ex31d.insereA(4, 2)
ex31d.insereA(4, 3)
ex31d.show()

print("O grafo é completo?", ex31d.isComplete())

print('\nNão Direcionado')

ex31nd = TGrafo(5)
ex31nd.insereA(0, 1)
ex31nd.insereA(1, 2)
ex31nd.insereA(2, 3)
ex31nd.show()

print("O grafo é completo?", ex31nd.isComplete())

ex31nd.insereA(0, 2)
ex31nd.insereA(0, 3)
ex31nd.insereA(0, 4)
ex31nd.insereA(1, 3)
ex31nd.insereA(1, 4)
ex31nd.insereA(2, 4)
ex31nd.insereA(3, 4)
ex31nd.show()

print("O grafo é completo?", ex31nd.isComplete())

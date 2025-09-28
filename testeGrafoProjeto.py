from GrafoProjeto import GrafoMatriz

print("Teste 1: Ler arquivo, show e degree")
grafo = GrafoMatriz(rotulado=True)
lista_adj = grafo.lerArquivoMatrizAdj("templatetxt.txt")
grafo.show()
print("Degree de 'Alice':", grafo.degree('Alice'))

print("\nTeste 2: insereA")
grafo.show()
grafo.insereA('Bob', 'Alice', 2.0)
print("Após insereA('Bob', 'Alice', 2.0), degree de 'Alice':", grafo.degree('Alice'))

grafo.show()

print("\nTeste 3: removeA")
grafo.removeA('Bob', 'Alice')
grafo.show()
print("Após removeA('Bob', 'Alice'), degree de 'Alice':", grafo.degree('Alice'))

print("\nTeste 4: inDegree e outDegree de 'Alice'")
print("inDegree de 'Alice':", grafo.inDegree('Alice'))
print("outDegree de 'Alice':", grafo.outDegree('Alice'))

print("\nTeste 5: isSource e isSorvedouro")
print("Alice é fonte?", grafo.isSource('Alice'))       # esperado: 0
print("Alice é sorvedouro?", grafo.isSorvedouro('Alice')) # esperado: 1

print("\nTeste 6: adicionarVertice")
try:
    grafo.adicionarVertice('Dave')
    print("Vértice 'Dave' adicionado com sucesso.")
except Exception as e:
    print("Erro ao adicionar 'Dave':", e)
grafo.show()

print("\nTeste 7: removerVertice")
try:
    grafo.removerVertice('Bob')
    print("Vértice 'Bob' removido com sucesso.")
except Exception as e:
    print("Erro ao remover 'Bob':", e)
grafo.show()

print("\nTeste 8: degree de 'Dave' (após remoção de 'Bob')")
try:
    grafo.insereA('Dave','Alice', 5.0)
    grafo.show()

    print("Degree de 'Dave':", grafo.degree('Dave'))
    print("Degree de 'Alice':", grafo.degree('Alice'))

except Exception as e:
    print("Erro ao acessar degree de 'Dave':", e)
    print("Isso indica que os índices internos do grafo não foram atualizados após a remoção de vértices.")


print("Teste Categoria 0 (grafo não conexo):")
grafo0 = GrafoMatriz(rotulado=True)
grafo0.adicionarVertice("A")
grafo0.adicionarVertice("B")
grafo0.adicionarVertice("C")

grafo0.insereA("A", "B", 1.0)
grafo0.show()
print("Categoria Conexidade (esperado 0):", grafo0.categoriaConexidade())
print()

print("Teste Categoria 3 (grafo fortemente conexo):")
grafo3 = GrafoMatriz(rotulado=True)
grafo3.adicionarVertice("A")
grafo3.adicionarVertice("B")
grafo3.adicionarVertice("C")

grafo3.insereA("A", "B", 1.0)
grafo3.insereA("B", "A", 1.0)
grafo3.insereA("A", "C", 1.0)
grafo3.insereA("C", "A", 1.0)
grafo3.insereA("B", "C", 1.0)
grafo3.insereA("C", "B", 1.0)
grafo3.show()
print("Categoria Conexidade (esperado 3):", grafo3.categoriaConexidade())
print()

print("Teste Categoria 2 (grafo com pelo menos uma aresta bidirecional):")
grafo2 = GrafoMatriz(rotulado=True)
grafo2.adicionarVertice("A")
grafo2.adicionarVertice("B")
grafo2.adicionarVertice("C")



grafo2.insereA("A", "B", 1.0)
grafo2.insereA("B", "A", 1.0)
grafo2.insereA("B", "C", 1.0)

grafo2.show()
print("Categoria Conexidade (esperado 2):", grafo2.categoriaConexidade())
print()

print("Teste Categoria 1 (grafo conectado sem arestas bidirecionais):")
grafo1 = GrafoMatriz(rotulado=True)
grafo1.adicionarVertice("A")
grafo1.adicionarVertice("B")
grafo1.adicionarVertice("C")


grafo1.insereA("A", "B", 1.0)
grafo1.insereA("B", "C", 1.0)
grafo1.show()
print("Categoria Conexidade (esperado 1):", grafo1.categoriaConexidade())


grafo.removerVertice('Dave')

grafo.gravarArquivoMatrizAdj("grafoResultado.txt")

grafo.show()
print("Dados gravados com sucesso.")


grafo = GrafoMatriz(rotulado=True)
grafo.adicionarVertice('A')
grafo.adicionarVertice('B')
grafo.adicionarVertice('C')
grafo.adicionarVertice('D')
grafo.adicionarVertice('E')
grafo.adicionarVertice('F')
grafo.adicionarVertice('G')

grafo.insereA('A', 'B')
grafo.insereA('B', 'C')
grafo.insereA('C', 'D')
grafo.insereA('D', 'E')
grafo.insereA('E', 'F')
grafo.insereA('F', 'G')
grafo.insereA('A', 'G')
grafo.insereA('C', 'F')
print("Conectividade Forte:", grafo.hConexidade())
print("Caminho Euleriano:", grafo.caminhoEuleriano())
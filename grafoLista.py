class GrafoLista:
    TAM_MAX_DEFAULT = 100

    def __init__(self, n=TAM_MAX_DEFAULT):
        self.n = n
        self.m = 0
        self.listaAdj = [[] for _ in range(self.n)]

    def insereA(self, v, w):
        if self.listaAdj[v] is None or self.listaAdj[w] is None:
            raise ValueError("Uma das extremidades da aresta foi removida e não pode receber arestas.")
        self.listaAdj[v].append(w)
        self.m += 1

    def removeA(self, v, w):
        if self.listaAdj[v] is not None and w in self.listaAdj[v]:
            self.listaAdj[v].remove(w)
            self.m -= 1

    def remover_vertice(self, v):
        if v >= len(self.listaAdj):
            raise ValueError('Vértice não existe.')

        for lista in self.listaAdj:
            if lista is not None and v in lista:
                lista.remove(v)

        self.listaAdj[v] = None
        self.n -= 1

    def inDegree(self, v):
        if self.listaAdj[v] is None:
            raise ValueError("Vértice foi removido e não tem grau de entrada.")
        grauVertice = 0
        for i in range(len(self.listaAdj)):
            if self.listaAdj[i] is not None and v in self.listaAdj[i]:
                grauVertice += 1
        return grauVertice

    def outDegree(self, v):
        if self.listaAdj[v] is None:
            raise ValueError("Vértice foi removido e não tem grau de saída.")
        return len(self.listaAdj[v])

    def degree(self, v):
        return self.inDegree(v) + self.outDegree(v)

    def isEqual(self, grafoA, grafoB):
        if len(grafoA.listaAdj) != len(grafoB.listaAdj):
            return False
        for i in range(len(grafoA.listaAdj)):
            if grafoA.listaAdj[i] is None or grafoB.listaAdj[i] is None:
                continue
            if set(grafoA.listaAdj[i]) != set(grafoB.listaAdj[i]):
                return False
        return True

    def isComplete(self):
        total_vertices = len(self.listaAdj)
        for v in range(len(self.listaAdj)):
            if len(self.listaAdj[v]) != total_vertices - 1:
                return False
        return True

    def listToMatrix(self):
        matrix = [[0 for _ in range(len(self.listaAdj))] for _ in range(len(self.listaAdj))]
        for v in range(len(self.listaAdj)):
            if self.listaAdj[v] is not None:
                for w in self.listaAdj[v]:
                    matrix[v][w] = 1
        return matrix

    def matrixToList(self, matrix):
        self.n = len(matrix)
        self.listaAdj = [[] for _ in range(self.n)]
        for v in range(self.n):
            for w in range(self.n):
                if matrix[v][w] != 0:
                    self.insereA(v, w)
        print("\nLista de adjacência:")
        for v in range(self.n):
            if self.listaAdj[v] is not None:
                print(f"{v}: {' '.join(map(str, self.listaAdj[v]))}")
        return self.listaAdj

    def show(self):
        for i in range(len(self.listaAdj)):
            if self.listaAdj[i] is not None:
                print(f"\n{i:2d}: ", end="")
                for val in self.listaAdj[i]:
                    print(f"{val:2d}", end="")
        print("\n\nfim da impressao do grafo.")


class TGrafo:
    def __init__(self, vertices):
        self.vertices = vertices
        self.lista_adj = {i: [] for i in range(vertices)}

    def insereA(self, u, v):
        self.lista_adj[u].append(v)
        self.lista_adj[v].append(u)

    def remover_vertice(self, vertice):
        if vertice in self.lista_adj:
            del self.lista_adj[vertice]
            for v in self.lista_adj:
                if vertice in self.lista_adj[v]:
                    self.lista_adj[v].remove(vertice)

    def isSource(self, vertice):
        grau_entrada = {i: 0 for i in range(self.vertices)}

        for v in self.lista_adj:
            for vizinho in self.lista_adj[v]:
                grau_entrada[vizinho] += 1

        if grau_entrada[vertice] == 0 and len(self.lista_adj[vertice]) > 0:
            return 1
        return 0

    def show(self):
        for vertice, vizinhos in self.lista_adj.items():
            print(f"{vertice}: {vizinhos}")

    def isSorvedouro(self, vertice):
        grau_entrada = {i: 0 for i in range(self.vertices)}

        for v in self.lista_adj:
            for vizinho in self.lista_adj[v]:
                grau_entrada[vizinho] += 1

        if grau_entrada[vertice] > 0 and len(self.lista_adj[vertice]) == 0:
            return 1
        return 0

    def isSimetric(self):
        for u in self.lista_adj:
            for v in self.lista_adj[u]:
                if u not in self.lista_adj[v]:
                    return 0
        return 1

    @classmethod
    def lerArquivo(cls, nome_arquivo):
        with open(nome_arquivo, 'r') as arquivo:
            vertices = int(arquivo.readline().strip())
            arestas = int(arquivo.readline().strip())
            grafo = cls(vertices)
            for _ in range(arestas):
                u, v = map(int, arquivo.readline().strip().split())
                grafo.insereA(u, v)
        return grafo

    def exibir_grafo(self):
        for vertice, vizinhos in self.lista_adj.items():
            print(f"{vertice}: {vizinhos}")

    def isComplete(self):
        total_vertices = len(self.lista_adj)
        for v in self.lista_adj:
            if len(self.lista_adj[v]) != total_vertices - 1:
                return False
            for w in self.lista_adj[v]:
                if v not in self.lista_adj[w]:
                    return False
        return True

    def inverter_adj(self):
        for v in range(len(self.lista_adj)):
            self.lista_adj[v].reverse()

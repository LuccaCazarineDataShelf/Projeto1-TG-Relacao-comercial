# -*- coding: utf-8 -*-

class GrafoMatriz:
    TAM_MAX_DEFAULT = 100  # qtde de vértices máxima default
    INF = float('inf')  # Valor para representar ausência de conexão em grafos com peso

    def __init__(self, n=TAM_MAX_DEFAULT, rotulado=False):
        self.n = n  # Quantidade de vértices
        self.m = 0  # Quantidade de arestas
        self.rotulado = rotulado  # Define se o grafo é rotulado
        self.nomes = {i: f"V{i}" for i in range(n)}  # Nome dos vértices
        valor_padrao = self.INF if rotulado else 0
        self.adj = [[valor_padrao for _ in range(n)] for _ in range(n)]  # Matriz de adjacência

    def insereA(self, vertice_origem, vertice_alvo, peso=1.0):
        if self.rotulado:
            if self.adj[vertice_origem][vertice_alvo] == self.INF:
                self.adj[vertice_origem][vertice_alvo] = peso
                self.m += 1
        else:
            if self.adj[vertice_origem][vertice_alvo] == 0:
                self.adj[vertice_origem][vertice_alvo] = 1
                self.m += 1

    def removeA(self, vertice_origem, vertice_alvo):
        if self.rotulado:
            if self.adj[vertice_origem][vertice_alvo] != self.INF:
                self.adj[vertice_origem][vertice_alvo] = self.INF
                self.m -= 1
        else:
            if self.adj[vertice_origem][vertice_alvo] == 1:
                self.adj[vertice_origem][vertice_alvo] = 0
                self.m -= 1

    def removeV(self, vertice):
        if isinstance(vertice, int):
            if vertice not in self.nomes or self.nomes[vertice] is None:
                raise ValueError("Vértice não existe.")
        else:
            if vertice not in self.nomes.values():
                raise ValueError("Vértice não existe.")
            vertice = list(self.nomes.keys())[list(self.nomes.values()).index(vertice)]

        self.nomes[vertice] = None
        for i in range(self.n):
            self.adj[vertice][i] = self.INF if self.rotulado else 0
            self.adj[i][vertice] = self.INF if self.rotulado else 0

    def inDegree(self, vertice):
        if isinstance(vertice, int):
            if vertice not in self.nomes or self.nomes[vertice] is None:
                raise ValueError("Vértice não existe.")
        else:
            if vertice not in self.nomes.values():
                raise ValueError("Vértice não existe.")
            vertice = list(self.nomes.keys())[list(self.nomes.values()).index(vertice)]

        return sum(1 for i in range(self.n) if self.adj[i][vertice] != self.INF and self.adj[i][vertice] != 0)

    def outDegree(self, vertice):
        if isinstance(vertice, int):
            if vertice not in self.nomes or self.nomes[vertice] is None:
                raise ValueError("Vértice não existe.")
        else:
            if vertice not in self.nomes.values():
                raise ValueError("Vértice não existe.")
            vertice = list(self.nomes.keys())[list(self.nomes.values()).index(vertice)]

        return sum(1 for j in range(self.n) if self.adj[vertice][j] != self.INF and self.adj[vertice][j] != 0)

    def degree(self, vertice):
        return self.inDegree(vertice) + self.outDegree(vertice)

    def show(self):
        nomes_ordenados = [i for i in sorted(self.nomes.keys()) if self.nomes[i] is not None]
        print("   " + " ".join(f"{self.nomes[i]:3}" for i in nomes_ordenados))
        for vertice in nomes_ordenados:
            linha = " ".join(f"{self.adj[vertice][j]:3}" for j in nomes_ordenados)
            print(f"{self.nomes[vertice]:2} | {linha}")

    def isSource(self, vertice):
        saida = self.outDegree(vertice)
        entrada = self.inDegree(vertice)

        if saida > 0 and entrada == 0:
            return 1
        else:
            return 0

    def isSorvedouro(self, vertice):
        saida = self.outDegree(vertice)
        entrada = self.inDegree(vertice)

        if saida == 0 and entrada > 0:
            return 1
        else:
            return 0

    def isComplete(self):
        n = len(self.adj)
        for i in range(n):
            for j in range(n):
                if i != j and (self.adj[i][j] == 0 or self.adj[i][j] == self.INF):
                    return False
        return True

    def complementar(self):
        grafo_comp = GrafoMatriz(self.n, self.rotulado)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    if not self.rotulado:
                        grafo_comp.adj[i][j] = 1 if self.adj[i][j] == 0 else 0
                    else:
                        if self.adj[i][j] == self.INF:
                            grafo_comp.adj[i][j] = 1
                        else:
                            grafo_comp.adj[i][j] = self.INF

        return grafo_comp
    
    def _tem_aresta(self, i: int, j: int) -> bool:
     return (self.adj[i][j] != self.INF) if self.rotulado else (self.adj[i][j] != 0)

    def _vizinhos(self, i: int):
        for j in range(self.n):
            if self._tem_aresta(i, j):
                yield j
    
    def _vizinhos_undirected(self, i: int):
     for j in range(self.n):
        if self._tem_aresta(i, j) or self._tem_aresta(j, i):
            yield j

    def _dfs_ordem(self, u: int, vis: list, pilha: list):
        vis[u] = True
        for v in self._vizinhos(u):
            if not vis[v]:
                self._dfs_ordem(v, vis, pilha)
        pilha.append(u)

    def _dfs_comp(self, u: int, vis: list, comp: list, Gt):
        vis[u] = True
        comp.append(u)
        for v in Gt._vizinhos(u):
            if not vis[v]:
                self._dfs_comp(v, vis, comp, Gt)

    def _transposto(self):
        Gt = GrafoMatriz(rotulado=self.rotulado)
        for i in range(self.n):
            Gt.adicionarVertice(self.indices[i])
        for i in range(self.n):
            for j in range(self.n):
                if self._tem_aresta(i, j):
                    if self.rotulado:
                        Gt.insereA(self.indices[j], self.indices[i], self.adj[i][j])
                    else:
                        Gt.insereA(self.indices[j], self.indices[i])
        return Gt

    def grafoReduzido(self):
        """Condensação por SCC: cada componente vira um nó; arestas entre SCCs distintas."""
        comp_id, comps = self.scc()
        nome_comp = [f"SCC_{i}" for i in range(len(comps))]

        Gc = GrafoMatriz(rotulado=False)  
        for rotulo in nome_comp:
            Gc.adicionarVertice(rotulo)

        arestas_c = set()
        for i in range(self.n):
            for j in range(self.n):
                if self._tem_aresta(i, j):
                    ci, cj = comp_id[i], comp_id[j]
                    if ci != cj:
                        arestas_c.add((ci, cj))

        for (ci, cj) in sorted(arestas_c):
            Gc.insereA(nome_comp[ci], nome_comp[cj])
        return Gc

    def lerArquivo(self, arquivo):
        with open(arquivo, 'r') as arquivo:
            self.n = int(arquivo.readline().strip())
            self.m = int(arquivo.readline().strip())

            self.nomes = {i: f"V{i}" for i in range(self.n)}
            valor_padrao = self.INF if self.rotulado else 0
            self.adj = [[valor_padrao for _ in range(self.n)] for _ in range(self.n)]

            for _ in range(self.m):
                u, v = map(int, arquivo.readline().strip().split())
                self.insereA(u, v)


    def simetrico(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.adj[i][j] != self.INF and self.adj[j][i] != self.INF:
                    if self.adj[i][j] != self.adj[j][i]:
                        return 0
                elif self.adj[i][j] != self.adj[j][i]:
                    return 0
        return 1

    def isConected(self):
        visitados = set()

        def verifica(v):
            visitados.add(v)
            for j in range(self.n):
                if self.adj[v][j] != 0 and self.adj[v][j] != self.INF and j not in visitados:
                    verifica(j)

        for v in range(self.n):
            if self.nomes[v] is not None:
                verifica(v)

                break

        for v in range(self.n):
            if self.nomes[v] is not None and v not in visitados:
                return 1
        return 0

    def dfs(self, vertice, visitados):
        visitados[vertice] = True
        for i in range(self.n):
            if self.adj[vertice][i] != self.INF and self.adj[vertice][i] != 0 and not visitados[i]:
                self.dfs(i, visitados)

    def categoriaConexidade(self):
        """C3: fortemente conexo; C2: unilateral; C1: fracamente conexo; C0: desconexo."""
        n = self.n
        if n == 0:
            return 0
        if n == 1:
            return 3
        vis = [False]*n
        ordem = []

        def tem_aresta(i, j):
            return (self.adj[i][j] != self.INF) if self.rotulado else (self.adj[i][j] != 0)

        def dfs1(u):
            vis[u] = True
            for v in range(n):
                if tem_aresta(u, v) and not vis[v]:
                    dfs1(v)
            ordem.append(u)

        for s in range(n):
            if not vis[s]:
                dfs1(s)

        vis = [False]*n
        comp_id = [-1]*n
        comps = []
        cid = 0

        def dfs2(u, cid):
            vis[u] = True
            comp = [u]
            stack = [u]
            while stack:
                x = stack.pop()
                for y in range(n):
                    if tem_aresta(y, x) and not vis[y]:
                        vis[y] = True
                        comp.append(y)
                        stack.append(y)
            return comp

        while ordem:
            u = ordem.pop()
            if not vis[u]:
                comp = dfs2(u, cid)
                for v in comp:
                    comp_id[v] = cid
                comps.append(comp)
                cid += 1

        if len(comps) == 1:
            return 3  # C3

        alcan = []
        for s in range(n):
            vis = [False]*n
            vis[s] = True
            stack = [s]
            while stack:
                u = stack.pop()
                for v in range(n):
                    if tem_aresta(u, v) and not vis[v]:
                        vis[v] = True
                        stack.append(v)
            alcan.append(vis)

        unilateral = True
        for u in range(n):
            for v in range(u+1, n):
                if not (alcan[u][v] or alcan[v][u]):
                    unilateral = False
                    break
            if not unilateral:
                break
        if unilateral:
            return 2  
        
        vis = [False]*n
        from collections import deque
        q = deque([0])
        vis[0] = True
        while q:
            u = q.popleft()
            for v in range(n):
                if (tem_aresta(u, v) or tem_aresta(v, u)) and not vis[v]:
                    vis[v] = True
                    q.append(v)

        if all(vis):
            return 1  
        return 0

    def dijkstra(self, origem):
        if not self.rotulado:
            raise ValueError("O algoritmo de Dijkstra requer grafos rotulados com pesos.")

        d = [self.INF] * self.n
        rot = [None] * self.n
        d[origem] = 0

        nao_visitados = {i for i in range(self.n) if self.nomes[i] is not None}

        while nao_visitados:
            u = min(nao_visitados, key=lambda x: d[x])

            if d[u] == self.INF:
                break

            nao_visitados.remove(u)

            for v in range(self.n):
                peso = self.adj[u][v]
                if v in nao_visitados and peso != self.INF:
                    nova_distancia = d[u] + peso
                    if nova_distancia < d[v]:
                        d[v] = nova_distancia
                        rot[v] = u

        return d, rot

    def prim(self, inicio=None):
        validos = [i for i, nome in self.nomes.items() if nome is not None]
        if not validos:
            return 0, []

        # traduz ponto de partida
        if inicio is None:
            u0 = validos[0]
        elif isinstance(inicio, str):
            # busca índice pelo nome
            rev = {v: k for k, v in self.nomes.items() if v is not None}
            if inicio not in rev:
                raise ValueError(f"Vértice '{inicio}' não existe.")
            u0 = rev[inicio]
        else:
            u0 = inicio
            if u0 not in validos:
                raise ValueError(f"Índice de vértice inválido: {u0}")

        visitados = {u0}
        mst = []
        custo_total = 0

        while len(visitados) < len(validos):
            menor = self.INF
            sel_u = sel_v = None

            for u in visitados:
                for v in validos:
                    if v in visitados:
                        continue
                    peso = self.adj[u][v]
                    if (self.rotulado and peso != self.INF or
                            not self.rotulado and peso != 0):
                        if peso < menor:
                            menor = peso
                            sel_u, sel_v = u, v

            if sel_v is None:
                # grafo desconexo
                break

            visitados.add(sel_v)
            custo_total += menor
            mst.append((self.nomes[sel_u], self.nomes[sel_v], menor))

        return custo_total, mst

class GrafoMatrizND:
    TAM_MAX_DEFAULT = 100
    INF = float('inf')

    def __init__(self, n=TAM_MAX_DEFAULT, rotulado=False):
        self.n = n  #
        self.m = 0  #
        self.rotulado = rotulado
        self.nomes = {i: f"V{i}" for i in range(n)}
        valor_padrao = self.INF if rotulado else 0
        self.adj = [[valor_padrao for _ in range(n)] for _ in range(n)]

    def insereA(self, vertice_origem, vertice_alvo, peso=1.0):
        if self.rotulado:
            if self.adj[vertice_origem][vertice_alvo] == self.INF:
                self.adj[vertice_origem][vertice_alvo] = peso
                self.adj[vertice_alvo][vertice_origem] = peso
                self.m += 1
        else:
            if self.adj[vertice_origem][vertice_alvo] == 0:
                self.adj[vertice_origem][vertice_alvo] = 1
                self.adj[vertice_alvo][vertice_origem] = 1
                self.m += 1

    def removeA(self, vertice_origem, vertice_alvo):
        if self.rotulado:
            if self.adj[vertice_origem][vertice_alvo] != self.INF:
                self.adj[vertice_origem][vertice_alvo] = self.INF
                self.adj[vertice_alvo][vertice_origem] = self.INF
                self.m -= 1
        else:
            if self.adj[vertice_origem][vertice_alvo] == 1:
                self.adj[vertice_origem][vertice_alvo] = 0
                self.adj[vertice_alvo][vertice_origem] = 0
                self.m -= 1

    def grau(self, vertice):
        if isinstance(vertice, int):
            if vertice not in self.nomes or self.nomes[vertice] is None:
                raise ValueError("Vértice não existe.")
        else:
            if vertice not in self.nomes.values():
                raise ValueError("Vértice não existe.")
            vertice = list(self.nomes.keys())[list(self.nomes.values()).index(vertice)]

        return sum(1 for j in range(self.n) if self.adj[vertice][j] != self.INF and self.adj[vertice][j] != 0)

    def removeV(self, vertice):
        if isinstance(vertice, int):
            if vertice not in self.nomes or self.nomes[vertice] is None:
                raise ValueError("Vértice não existe.")
        else:
            if vertice not in self.nomes.values():
                raise ValueError("Vértice não existe.")
            vertice = list(self.nomes.keys())[list(self.nomes.values()).index(vertice)]

        self.nomes[vertice] = None
        for i in range(self.n):
            self.adj[vertice][i] = 0
            self.adj[i][vertice] = 0
        self.m = sum(
            1 for i in range(self.n) for j in range(i, self.n) if self.adj[i][j] != 0 and self.adj[i][j] != self.INF)

    def completo(self):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.nomes[i] is not None and self.nomes[j] is not None:
                    if self.adj[i][j] == 0 or self.adj[i][j] == self.INF:
                        return False
        return True

    def show(self):
        nomes_ordenados = [i for i in sorted(self.nomes.keys()) if self.nomes[i] is not None]
        print("   " + " ".join(f"{self.nomes[i]:3}" for i in nomes_ordenados))
        for vertice in nomes_ordenados:
            linha = " ".join(f"{self.adj[vertice][j]:3}" for j in nomes_ordenados)
            print(f"{self.nomes[vertice]:2} | {linha}")

    def complementar(self):
        grafoComp = GrafoMatrizND(self.n, self.rotulado)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    if not self.rotulado:
                        grafoComp.adj[i][j] = 1 if self.adj[i][j] == 0 else 0
                    else:
                        if self.adj[i][j] == self.INF:
                            grafoComp.adj[i][j] = 1
                        else:
                            grafoComp.adj[i][j] = self.INF
        return grafoComp

    def prim(self, inicio=None):
        validos = [i for i, nome in self.nomes.items() if nome is not None]
        if not validos:
            return 0, []

        if inicio is None:
            u0 = validos[0]
        elif isinstance(inicio, str):
            rev = {v: k for k, v in self.nomes.items() if v is not None}
            if inicio not in rev:
                raise ValueError(f"Vértice '{inicio}' não existe.")
            u0 = rev[inicio]
        else:
            u0 = inicio
            if u0 not in validos:
                raise ValueError(f"Índice de vértice inválido: {u0}")

        visitados = {u0}
        mst = []
        custo_total = 0

        while len(visitados) < len(validos):
            menor = self.INF
            sel_u = sel_v = None

            for u in visitados:
                for v in validos:
                    if v in visitados:
                        continue
                    peso = self.adj[u][v]
                    if (self.rotulado and peso != self.INF or
                            not self.rotulado and peso != 0):
                        if peso < menor:
                            menor = peso
                            sel_u, sel_v = u, v

            if sel_v is None:
                # desconexo
                break

            visitados.add(sel_v)
            custo_total += menor
            mst.append((self.nomes[sel_u], self.nomes[sel_v], menor))

        return custo_total, mst


    def coloracao(self):
        validos = [i for i, nome in self.nomes.items() if nome is not None]
        classes = []
        for u in validos:
            k = 0
            while True:
                if k >= len(classes):
                    classes.append(set())
                vizinhos = [v for v in validos if self.adj[u][v] != 0 and self.adj[u][v] != self.INF]
                if not any(v in classes[k] for v in vizinhos):
                    classes[k].add(u)
                    break
                k += 1
        return [{self.nomes[v] for v in classe} for classe in classes]
    
    def scc(self):
        """Retorna (comp_id, componentes) via Kosaraju.
        comp_id[i] = id da SCC do vértice i; componentes = lista de listas de índices."""
        if self.n == 0:
            return [], []

        vis = [False]*self.n
        pilha = []
        for i in range(self.n):
            if not vis[i]:
                self._dfs_ordem(i, vis, pilha)

        Gt = self._transposto()
        vis = [False]*self.n
        comp_id = [-1]*self.n
        componentes = []
        cid = 0

        while pilha:
            u = pilha.pop()
            if not vis[u]:
                comp = []
                Gt._dfs_comp(u, vis, comp, Gt)
                for v in comp:
                    comp_id[v] = cid
                componentes.append(comp)
                cid += 1

        return comp_id, componentes

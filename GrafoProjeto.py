from collections import deque

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx


class GrafoMatriz:
    TAM_MAX_DEFAULT = 10000
    INF = float('inf')

    def __init__(self, rotulado=False):
        self.n = 0
        self.m = 0
        self.rotulado = rotulado
        self.indices = {}  # mapeia índice -> nome
        self.nomes = {}    # mapeia nome -> índice
        self.adj = []

    def adicionarVertice(self, nome):
        if nome in self.nomes:
            raise ValueError("Vértice já existe.")

        self.indices[self.n] = nome
        self.nomes[nome] = self.n
        self.n += 1

        for linha in self.adj:
            linha.append(self.INF if self.rotulado else 0)
        self.adj.append([self.INF if self.rotulado else 0] * self.n)


    def insereA(self, origem, destino, peso=1.0):
        if origem not in self.nomes or destino not in self.nomes:
            raise ValueError("Vértice não encontrado.")

        index_origem = self.nomes[origem]
        index_destino = self.nomes[destino]

        if self.rotulado:
            if self.adj[index_origem][index_destino] == self.INF:
                self.adj[index_origem][index_destino] = peso
                self.m += 1
        else:
            if self.adj[index_origem][index_destino] == 0:
                self.adj[index_origem][index_destino] = 1
                self.m += 1

    def removerVertice(self, vertice):
        if vertice not in self.nomes:
            raise ValueError("Vértice não encontrado.")

        index_removido = self.nomes[vertice]

        # Remove o vértice dos mapeamentos originais
        del self.nomes[vertice]
        old_indices = self.indices.copy()
        del old_indices[index_removido]

        # Remove a linha correspondente e a coluna de cada linha
        del self.adj[index_removido]
        for linha in self.adj:
            del linha[index_removido]
        self.n -= 1

        # Reconstroi os mapeamentos de índices e nomes com índices de 0 a n-1
        new_indices = {}
        new_nomes = {}
        remaining = [old_indices[k] for k in sorted(old_indices.keys())]
        for novo_indice, nome in enumerate(remaining):
            new_indices[novo_indice] = nome
            new_nomes[nome] = novo_indice

        self.indices = new_indices
        self.nomes = new_nomes

        # Recalcula o número de arestas após a remoção
        novo_m = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.rotulado:
                    if self.adj[i][j] != self.INF:
                        novo_m += 1
                else:
                    if self.adj[i][j] != 0:
                        novo_m += 1
        self.m = novo_m
        self.indices = new_indices
        self.nomes = new_nomes

    def removeA(self, origem, destino):
        if origem not in self.nomes or destino not in self.nomes:
            raise ValueError("Vértice não encontrado.")

        index_origem = self.nomes[origem]
        index_destino = self.nomes[destino]

        if self.rotulado:
            if self.adj[index_origem][index_destino] != self.INF:
                self.adj[index_origem][index_destino] = self.INF
                self.m -= 1
        else:
            if self.adj[index_origem][index_destino] == 1:
                self.adj[index_origem][index_destino] = 0
                self.m -= 1

    def inDegree(self, vertice):
        if vertice not in self.nomes:
            raise ValueError("Vértice não encontrado.")
        index = self.nomes[vertice]
        return sum(1 for i in range(self.n) if self.adj[i][index] != (self.INF if self.rotulado else 0))

    def outDegree(self, vertice):
        if vertice not in self.nomes:
            raise ValueError("Vértice não encontrado.")
        index = self.nomes[vertice]
        return sum(1 for i in range(self.n) if self.adj[index][i] != (self.INF if self.rotulado else 0))

    def degree(self, vertice):
        return self.inDegree(vertice) + self.outDegree(vertice)

    def isSource(self, vertice):
        saida = self.outDegree(vertice)
        entrada = self.inDegree(vertice)
        return 1 if (saida > 0 and entrada == 0) else 0

    def isSorvedouro(self, vertice):
        saida = self.outDegree(vertice)
        entrada = self.inDegree(vertice)
        return 1 if (saida == 0 and entrada > 0) else 0
    
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

    def show(self):
        print("\nMatriz de Adjacência:")
        nomes_ordenados = [self.nomes_inv(i) for i in range(self.n)]
        print("   " + " ".join(f"{nome:3}" for nome in nomes_ordenados))
        print("   " + "---" * self.n)
        for i, nome in enumerate(nomes_ordenados):
            linha = " ".join(f"{self.adj[i][j]:3}" for j in range(self.n))
            print(f"{nome:2} | {linha}")

    def nomes_inv(self, indice):
        return self.indices.get(indice, "?")

    def lerArquivoMatrizAdj(self, arquivo):

        with open(arquivo, 'r') as arq:
            tipo = arq.readline().strip()
            if tipo != "6":
                print("Tipo de grafo não é compativel")
                return None

            num_vertices = int(arq.readline().strip())
            for _ in range(num_vertices):
                linha = arq.readline().strip()
                partes = linha.split(maxsplit=1)
                if len(partes) < 2:
                    continue

                nome_vertice = partes[1]
                self.adicionarVertice(nome_vertice)

            num_arestas = int(arq.readline().strip())
            for _ in range(num_arestas):
                linha = arq.readline().strip()
                partes = linha.split()
                if len(partes) != 3:
                    continue
                origem_idx = int(partes[0])
                destino_idx = int(partes[1])
                try:
                    peso = float(partes[2])
                except ValueError:
                    peso = partes[2]

                origem_nome = self.indices.get(origem_idx)
                destino_nome = self.indices.get(destino_idx)
                if origem_nome is None or destino_nome is None:
                    print(f"Índices inválidos na aresta: {linha}")
                    continue
                self.insereA(origem_nome, destino_nome, peso)
        return self.adj

    def gravarArquivoMatrizAdj(self, arquivo):

        with open(arquivo, 'w') as arq:
            arq.write("6\n")
            arq.write(f"{self.n}\n")
            for i in range(self.n):
                nome = self.indices[i]
                arq.write(f"{i} {nome}\n")
            arq.write(f"{self.m}\n")
            for i in range(self.n):
                for j in range(self.n):
                    if self.rotulado:
                        if self.adj[i][j] != self.INF:
                            arq.write(f"{i} {j} {self.adj[i][j]}\n")
                    else:
                        if self.adj[i][j] != 0:
                            arq.write(f"{i} {j} {self.adj[i][j]}\n")

    def dfs(self, v, visitados):
        visitados[v] = True
        for i in range(self.n):
            if not visitados[i]:
                if self.rotulado:
                    if self.adj[v][i] != self.INF:
                        self.dfs(i, visitados)
                else:
                    if self.adj[v][i] != 0:
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

    def visitarNo(self, v, ordem_visita):

        print(f"Visitado: {self.indices[v]}")
        ordem_visita.append(self.indices[v])

    def noAdjacente(self, n, visitados):

        for i in range(self.n):
            if not visitados[i]:
                if self.rotulado:
                    if self.adj[n][i] != self.INF:
                        return i
                else:
                    if self.adj[n][i] != 0:
                        return i
        return -1

    def percursoProfundidade(self, vInicio):

        if isinstance(vInicio, str):
            if vInicio not in self.nomes:
                raise ValueError("Vértice não encontrado.")
            vInicio = self.nomes[vInicio]

        visitados = [False] * self.n
        pilha = []
        ordem_visita = []

        self.visitarNo(vInicio, ordem_visita)
        visitados[vInicio] = True
        pilha.append(vInicio)

        while pilha:
            n = pilha.pop()
            m = self.noAdjacente(n, visitados)
            while m != -1:
                self.visitarNo(m, ordem_visita)
                pilha.append(n)
                visitados[m] = True
                n = m
                m = self.noAdjacente(n, visitados)
        return ordem_visita

    def percursoLargura(self, vInicio):

        if isinstance(vInicio, str):
            if vInicio not in self.nomes:
                raise ValueError("Vértice não encontrado.")
            vInicio = self.nomes[vInicio]

        visitados = [False] * self.n
        fila = deque()
        ordem_visita = []

        self.visitarNo(vInicio, ordem_visita)
        visitados[vInicio] = True
        fila.append(vInicio)

        while fila:
            n = fila.popleft()

            m = self.noAdjacente(n, visitados)
            while m != -1:
                self.visitarNo(m, ordem_visita)
                visitados[m] = True
                fila.append(m)

                m = self.noAdjacente(n, visitados)

        return ordem_visita
    

    def listToMatrix(self):
        matrix = [[0 for _ in range(len(self.listaAdj))] for _ in range(len(self.listaAdj))]
        for v in range(len(self.listaAdj)):
            if self.listaAdj[v] is not None:
                for w in self.listaAdj[v]:
                    matrix[v][w] = 1
        self.matrizAdj = matrix
        return matrix

    def matrixToList(self):
        self.listaAdj = [[] for _ in range(self.n)]
        
        for i in range(self.n):
            for j in range(self.n):
                if self.rotulado:
                    if self.adj[i][j] != self.INF:
                        self.listaAdj[i].append(j)
                else:
                    if self.adj[i][j] != 0:
                        self.listaAdj[i].append(j)
        
        print("\nLista de adjacência:")
        for v in range(self.n):
            if self.listaAdj[v] is not None:
                print(f"{self.indices[v]}: {' '.join(map(str, self.listaAdj[v]))}")
        return self.listaAdj

    def prim(self, inicio=None):
        validos = list(range(self.n))
        if not validos:
            return 0, []

        # determina o vértice inicial
        if inicio is None:
            u0 = validos[0]
        elif isinstance(inicio, str):
            if inicio not in self.nomes:
                raise ValueError(f"Vértice '{inicio}' não existe.")
            u0 = self.nomes[inicio]
        else:
            u0 = inicio
            if u0 not in validos:
                raise ValueError(f"Índice de vértice inválido: {u0}")

        visitados = {u0}
        mst = []
        custo_total = 0

        # constrói a MST
        while len(visitados) < self.n:
            menor = self.INF
            sel_u = sel_v = None

            for u in visitados:
                for v in validos:
                    if v in visitados:
                        continue
                    peso = self.adj[u][v]
                    # verifica existência de aresta
                    if peso != (self.INF if self.rotulado else 0):
                        if peso < menor:
                            menor = peso
                            sel_u, sel_v = u, v

            # grafo desconexo?
            if sel_v is None:
                break

            visitados.add(sel_v)
            custo_total += menor
            # adiciona aresta pelo nome dos vértices
            mst.append((self.indices[sel_u], self.indices[sel_v], menor))

        return custo_total, mst

    def dijkstra(self, origem):
        if not self.rotulado:
            raise ValueError("O algoritmo de Dijkstra requer grafos rotulados com pesos.")

        # lista de vértices válidos
        validos = [i for i, nome in self.nomes.items() if nome is not None]

        # traduz origem
        if isinstance(origem, str):
            rev = {v: k for k, v in self.nomes.items() if v is not None}
            if origem not in rev:
                raise ValueError(f"Vértice '{origem}' não existe.")
            src = rev[origem]
        else:
            src = origem
            if src not in validos:
                raise ValueError(f"Índice de vértice inválido: {src}")

        # inicialização
        d = {i: self.INF for i in validos}
        pred = {i: None for i in validos}
        d[src] = 0
        nao_visitados = set(validos)

        while nao_visitados:
            u = min(nao_visitados, key=lambda x: d[x])
            if d[u] == self.INF:
                break
            nao_visitados.remove(u)

            for v in validos:
                peso = self.adj[u][v]
                if v in nao_visitados and peso != self.INF:
                    nova_dist = d[u] + peso
                    if nova_dist < d[v]:
                        d[v] = nova_dist
                        pred[v] = u

        # mapeia resultados para nomes
        distancias = {self.nomes[i]: d[i] for i in validos}
        predecessores = {self.nomes[i]: (self.nomes[pred[i]] if pred[i] is not None else None) for i in validos}

        return distancias, predecessores
    

    def hConexidade(self):
        # Verifica a conectividade no grafo original
        visitados = [False] * self.n
        self.dfs(0, visitados)
        if not all(visitados):
            return False

        # Cria o grafo transposto (com todas as arestas invertidas)
        grafo_transposto = GrafoMatriz(rotulado=self.rotulado)
        for vertice in self.nomes:
            grafo_transposto.adicionarVertice(vertice)
        
        for i in range(self.n):
            for j in range(self.n):
                if self.rotulado:
                    if self.adj[i][j] != self.INF:
                        grafo_transposto.insereA(self.indices[i], self.indices[j], self.adj[i][j])
                else:
                    if self.adj[i][j] != 0:
                        grafo_transposto.insereA(self.indices[i], self.indices[j])

        # Verifica a conectividade no grafo transposto
        visitados = [False] * self.n
        grafo_transposto.dfs(0, visitados)
        if not all(visitados):
            return False

        return True
    

    def caminhoEuleriano(self):
        visitados = [False] * self.n
        self.dfs(0, visitados)

        if not all(visitados):
            return False

        origem, destino = 0, 0 

        for i in range(self.n):
            in_degree = self.inDegree(self.indices[i])
            out_degree = self.outDegree(self.indices[i])
            if in_degree != out_degree:

                if in_degree - out_degree == 1:
                    origem += 1

                elif out_degree - in_degree == 1:
                    destino += 1

                else:
                    return False

        return (origem == 1 and destino == 1) or (origem == 0 and destino == 0)
    
    def imprimirArquivoLegivel(self, arquivo: str) -> None:
        """Lê um arquivo .txt no formato do projeto e imprime de forma legível."""
        def descricao_tipo(t: str) -> str:
            mapa = {
                "0": "Não direcionado, sem peso",
                "1": "Não direcionado, peso nas arestas",
                "2": "Não direcionado, peso nos vértices",
                "3": "Não direcionado, peso em vértices e arestas",
                "4": "Direcionado, sem peso",
                "5": "Direcionado, peso nos vértices",
                "6": "Direcionado, peso nas arestas",
                "7": "Direcionado, peso em vértices e arestas",
            }
            return mapa.get(str(t), "Tipo desconhecido")

        try:
            with open(arquivo, "r", encoding="utf-8") as f:
                tipo = f.readline().strip()
                if not tipo:
                    print("Arquivo vazio.")
                    return

                n = int(f.readline().strip())
                vertices = []  
                for _ in range(n):
                    linha = f.readline().rstrip("\n")
                    if not linha:
                        continue
                    partes = linha.split(maxsplit=1)  
                    idx = int(partes[0])
                    nome = partes[1] if len(partes) > 1 else ""
                    vertices.append((idx, nome))

                m = int(f.readline().strip())
                arestas = []  
                for _ in range(m):
                    linha = f.readline().strip()
                    if not linha:
                        continue
                    p = linha.split()
                    if len(p) < 2:
                        continue
                    u, v = int(p[0]), int(p[1])
                    w = None
                    if len(p) >= 3:
                        try:
                            w_val = float(p[2])
                            w = int(w_val) if w_val.is_integer() else w_val
                        except ValueError:
                            w = p[2]
                    arestas.append((u, v, w))

        except FileNotFoundError:
            print(f"Erro: arquivo '{arquivo}' não encontrado.")
            return
        except Exception as e:
            print(f"Erro ao ler '{arquivo}': {e}")
            return

        eh_direcionado = str(tipo) in {"4", "5", "6", "7"}
        seta = "->" if eh_direcionado else "--"

        idx_w = max(2, len(str(max((i for i, _ in vertices), default=0))))
        nome_w = max(5, max((len(lbl) for _, lbl in vertices), default=0))

        print(f"\nArquivo: {arquivo}")
        print(f"Tipo: {tipo} – {descricao_tipo(tipo)}")

        print(f"\nVértices (n={len(vertices)}):")
        print("  " + f"{'ID':>{idx_w}}  Nome")
        print("  " + "-" * idx_w + "  " + "-" * max(4, nome_w))
        for idx, label in vertices:
            print(f"  {idx:>{idx_w}}  {label}")

        print(f"\nArestas (m={len(arestas)}):")
        if arestas:
            print("  " + f"{'Orig':>{idx_w}}  {seta}  " + f"{'Dest':>{idx_w}}  Peso")
            print("  " + "-" * idx_w + "     " + "-" * idx_w + "  ----")
            for u, v, w in arestas:
                peso_str = "" if w is None else str(w)
                print(f"  {u:>{idx_w}}  {seta}  {v:>{idx_w}}  {peso_str}")
        else:
            print("  (sem arestas)")

    def listarGraus(self):
        for nome in self.nomes:
            grau = self.degree(nome)
            print(f"Vértice: {nome}, Grau: {grau}")

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

    def plotarGrafo(
        self,
        outfile: str = "grafo.png",
        layout: str = "auto",          
        show_weights: bool = False,    
        max_label_len: int = 16,      
    ):
        import matplotlib
        import matplotlib.pyplot as plt
        import networkx as nx
        import math

        N = max(1, self.n)
        G = nx.DiGraph() if self.rotulado else nx.Graph()

        def short(s: str) -> str:
            s = (s or "").strip()
            if len(s) <= max_label_len:
                return s
            return s[: max_label_len - 1] + "…"

        idx2label = {i: self.indices[i] for i in range(self.n)}
        short_label = {i: short(idx2label[i]) for i in range(self.n)}

        for i in range(self.n):
            G.add_node(i, label=short_label[i], full=idx2label[i])

        have_weight = False
        for i in range(self.n):
            for j in range(self.n):
                if self.rotulado:
                    w = self.adj[i][j]
                    if w != self.INF:
                        G.add_edge(i, j, weight=w)
                        have_weight = True
                else:
                    if self.adj[i][j] != 0:
                        G.add_edge(i, j)

        pos = None
        if layout == "auto":
            if N >= 40:
                try:
                    from networkx.drawing.nx_agraph import graphviz_layout
                    pos = graphviz_layout(G, prog="sfdp")
                except Exception:
                    pos = None
            if pos is None:
                try:
                    pos = nx.kamada_kawai_layout(G)
                except Exception:
                    pos = None
            if pos is None:
                k = 1.3 / math.sqrt(N)   
                pos = nx.spring_layout(G, seed=42, k=k, iterations=100)
        elif layout == "spring":
            k = 1.3 / math.sqrt(N)
            pos = nx.spring_layout(G, seed=42, k=k, iterations=100)
        elif layout == "kk":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "sfdp":
            from networkx.drawing.nx_agraph import graphviz_layout
            pos = graphviz_layout(G, prog="sfdp")
        else:
            pos = nx.spring_layout(G, seed=42)

        font_size = 12 if N <= 20 else max(6, 12 - 0.15 * (N - 20))
        node_size = 3000 if N <= 20 else max(600, int(3000 * (20 / N) ** 0.6))
        arrowsize = 12 if N <= 40 else 8
        edge_alpha = 0.6

        connectionstyle = "arc3,rad=0.15" if isinstance(G, nx.DiGraph) else None

        w = 10 if N <= 25 else (14 if N <= 60 else 18)
        h = w * 0.8
        plt.figure(figsize=(w, h))
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="skyblue")
        nx.draw_networkx_labels(
            G, pos, labels={i: G.nodes[i]["label"] for i in G.nodes},
            font_size=font_size, font_weight="bold"
        )
        nx.draw_networkx_edges(
            G, pos, alpha=edge_alpha, arrows=isinstance(G, nx.DiGraph),
            arrowsize=arrowsize, connectionstyle=connectionstyle, width=1.0, edge_color="gray"
        )

        if self.rotulado and have_weight and show_weights:
            if isinstance(show_weights, bool):
                max_labels = math.inf
            else:
                max_labels = int(show_weights)

            edge_labels = nx.get_edge_attributes(G, "weight")
            if max_labels != math.inf and len(edge_labels) > max_labels:
                try:
                    importantes = sorted(edge_labels.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:max_labels]
                except Exception:
                    importantes = list(edge_labels.items())[:max_labels]
                edge_labels = dict(importantes)

            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, font_size=max(6, font_size - 2), label_pos=0.5
            )

        plt.title("Grafo")
        plt.tight_layout()

        import matplotlib
        if matplotlib.get_backend().lower().endswith("agg"):
            plt.savefig(outfile, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Figura salva em '{outfile}' (layout={layout}).")
        else:
            plt.show()





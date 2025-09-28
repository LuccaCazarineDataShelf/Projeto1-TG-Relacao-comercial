from grafoMatriz import GrafoMatriz


def mostrar_caminhos_completos(origem, rot, distancias):
    for destino in range(len(distancias)):
        if distancias[destino] == float('inf'):
            print(f"Cidade {origem+1} → Cidade {destino+1}: não alcançável")
            continue

        caminho = []
        atual = destino
        while atual is not None:
            caminho.append(atual)
            atual = rot[atual]
        caminho.reverse()

        caminho_str = " → ".join(f"Cidade {v+1}" for v in caminho)
        print(f"{caminho_str} | Distância: {distancias[destino]}")

g = GrafoMatriz(4, rotulado=True)

# Arestas da linha 1 (vértice 0)
g.insereA(0, 1, 20)
g.insereA(0, 2, 30)
g.insereA(0, 3, 50)

# Arestas da linha 2 (vértice 1)
g.insereA(1, 0, 20)
g.insereA(1, 2, 40)
g.insereA(1, 3, 15)

# Arestas da linha 3 (vértice 2)
g.insereA(2, 0, 30)
g.insereA(2, 1, 40)
g.insereA(2, 3, 15)

# Arestas da linha 4 (vértice 3)
g.insereA(3, 0, 50)
g.insereA(3, 1, 15)
g.insereA(3, 2, 15)

distancias, predecessores = g.dijkstra(2)

print("Distâncias mínimas:", distancias)
print("Rota (predecessores):", predecessores)

origem = 2
distancias, rot = g.dijkstra(origem)

mostrar_caminhos_completos(origem, rot, distancias)
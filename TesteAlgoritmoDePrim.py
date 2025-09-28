from grafoMatriz import GrafoMatrizND, GrafoMatriz

prim = GrafoMatrizND(n=9, rotulado=True)

prim.insereA(0, 1, 4)  # 1–2 peso 4
prim.insereA(1, 2, 7)  # 2–3 peso 7
prim.insereA(2, 3, 5)  # 3–4 peso 5
prim.insereA(3, 4, 3)  # 4–5 peso 3
prim.insereA(4, 7, 4)  # 5–8 peso 4
prim.insereA(7, 8, 8)  # 8–9 peso 8
prim.insereA(8, 5, 5)  # 9–6 peso 5
prim.insereA(5, 0, 5)  # 6–1 peso 5

prim.insereA(5, 1, 3)  # 6–2 peso 3
prim.insereA(6, 2, 6)  # 7–3 peso 6
prim.insereA(6, 4, 2)  # 7–5 peso 2
prim.insereA(6, 5, 7)  # 7–6 peso 7
prim.insereA(6, 7, 6)  # 7–8 peso 6

prim.show()

custo, mst = prim.prim()
print("Custo total da MST:", custo)
print("Arestas da MST:")
for u, v, peso in mst:
    print(f"{u} – {v} (peso {peso})")

prim2 = GrafoMatriz(4, rotulado=True)

# Arestas da linha 1 (vértice 0)
prim2.insereA(0, 1, 20)
prim2.insereA(0, 2, 30)
prim2.insereA(0, 3, 50)

# Arestas da linha 2 (vértice 1)
prim2.insereA(1, 0, 20)
prim2.insereA(1, 2, 40)
prim2.insereA(1, 3, 15)

# Arestas da linha 3 (vértice 2)
prim2.insereA(2, 0, 30)
prim2.insereA(2, 1, 40)
prim2.insereA(2, 3, 15)

# Arestas da linha 4 (vértice 3)
prim2.insereA(3, 0, 50)
prim2.insereA(3, 1, 15)
prim2.insereA(3, 2, 15)

custo, mst = prim2.prim()
print("Custo total da MST:", custo)
print("Arestas da MST:")                 
for u, v, peso in mst:
    print(f"{u} – {v} (peso {peso})")
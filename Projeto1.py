from collections import deque
from GrafoProjeto import GrafoMatriz

TITULO_APLICACAO = "Rotas Comerciais entre Países – Direcionado com Pesos"  

grafo = GrafoMatriz(rotulado=True)

def mostrarTitulo(grafo):
    dirigido = "Direcionado"
    pesos = "com Pesos nas Arestas" if getattr(grafo, "rotulado", False) else "sem Pesos"
    largura = 72
    print("\n" + "=" * largura)
    print(TITULO_APLICACAO.center(largura))
    print(f"({dirigido}, {pesos})".center(largura))
    print("=" * largura)

def criarArquivo():
    arquivoGrafo = input("Insira o nome do arquivo de entrada (com extensão .txt): ")
    return arquivoGrafo

def criarNome():
    nome = input("Insira o nome do vértice: ")
    return nome

def switch(numeroMenu):
    if numeroMenu == 1:
        arquivo = criarArquivo()
        try:
            grafo.lerArquivoMatrizAdj(arquivo)
        except FileNotFoundError:
            print(f"Erro: O arquivo '{arquivo}' não foi encontrado.")
        except Exception as e:
            print(f"Ocorreu um erro ao tentar ler o arquivo: {e}")
        
    elif numeroMenu == 2:
        if grafo.n > 0:
            arquivo = criarArquivo()
            try:
                grafo.gravarArquivoMatrizAdj(arquivo)
            except Exception as e:
                print(f"Ocorreu um erro ao tentar gravar o arquivo: {e}")
        else:
            print("O grafo ainda não foi criado. Por favor, crie um grafo primeiro.")
            switch(numeroMenu)

    elif numeroMenu == 3:
        nome = criarNome()
        try:
            grafo.adicionarVertice(nome)
            print(f"Vértice '{nome}' adicionado com sucesso.")
        except ValueError as e:
            print(e)
        grafo.show()

    elif numeroMenu == 4:
        origem = input("Insira o valor da origem: ")
        destino = input("Insira o valor do destino: ")
        peso = input("Insira o valor do peso (pressione Enter para usar o valor padrão 1.0): ")

        if peso == "":
            peso = 1.0
        else:
            try:
                peso = float(peso)
            except ValueError:
                print("Valor de peso inválido. Usando o valor padrão de 1.0.")
                peso = 1.0

        try:
            grafo.insereA(origem, destino, peso)
            grafo.show()
        except ValueError as e:
            print(f"Erro ao inserir a aresta: {e}")

    elif numeroMenu == 5:
        nome = criarNome()
        try:
            grafo.removerVertice(nome)
            print(f"Vértice '{nome}' removido com sucesso.")
        except ValueError as e:
            print(e)
        grafo.show()

    elif numeroMenu == 6:
        origem = input("Insira o valor da origem: ")
        destino = input("Insira o valor do destino: ")
        try:
            grafo.removeA(origem, destino)
            grafo.show()
        except ValueError as e:
            print(f"Erro ao remover aresta: {e}")

    elif numeroMenu == 7:
        reduzido = grafo.grafoReduzido()
        print("\n=== Grafo Reduzido (SCC condensation) ===")
        reduzido.show()

    elif numeroMenu == 8:
        tipoGrafo = input("Se quiser visualizar como lista digite a, se preferir matriz digite b: ")
        if tipoGrafo == "a":
            grafo.matrixToList()
        elif tipoGrafo == "b":
            grafo.show()
        else:
            print("Opção inválida. Tentando novamente.")
            numeroMenu = 8
            switch(numeroMenu)

    elif numeroMenu == 9:
        if grafo.n == 0:
         print("Grafo vazio. Use a opção 1 para carregar um arquivo primeiro.")
        cat = grafo.categoriaConexidade()
        desc = {
            3: "C3 (fortemente conexo)",
            2: "C2 (unilateral)",
            1: "C1 (fracamente conexo)",
            0: "C0 (desconexo)"
        }[cat]
        print(f"Conectividade do grafo direcionado: {desc}")

    elif numeroMenu == 10:
        if grafo.hConexidade():
            print("O grafo é h-conexo.")
        else:
            print("O grafo não é conexo.")
        
    elif numeroMenu == 11:
        custo, mst = grafo.prim()
        if custo == 0:
            print("O grafo não é conexo para gerar uma árvore geradora mínima.")
        else:
            print(f"Custo da árvore geradora mínima: {custo}")
            for u, v, peso in mst:
                print(f"Aresta: {u} - {v} | Peso: {peso}")
        
    elif numeroMenu == 12:
        grafo.listarGraus()


    elif numeroMenu == 13:
        if grafo.caminhoEuleriano():
            print("O grafo possui um caminho euleriano.")
        else:
            print("O grafo não possui um caminho euleriano.")

    elif numeroMenu == 14:
        nome = input("Nome do arquivo de imagem (ex.: grafo.png) [ENTER = grafo.png]: ").strip()
        if not nome:
            nome = "grafo.png"
        try:
            grafo.plotarGrafo(nome)
        except Exception as e:
            print(f"Erro ao plotar: {e}")

    elif numeroMenu == 15:
        arquivo = criarArquivo()
        try:
            grafo.imprimirArquivoLegivel(arquivo)
        except Exception as e:
            print(f"Ocorreu um erro ao exibir o arquivo: {e}")

    elif numeroMenu == 16:
        if grafo.n == 0:
            print("Grafo vazio. Use a opção 1 para carregar um arquivo primeiro.")
            return

        print("\n=== Aplicando técnicas ao problema de Rotas Comerciais ===")

        origem = input("Informe o país de ORIGEM para o caminho mínimo: ")
        destino = input("Informe o país de DESTINO para o caminho mínimo: ")

        try:
            distancias, predecessores = grafo.dijkstra(origem)
            if destino not in distancias or distancias[destino] == grafo.INF:
                print(f"\n[Técnica 1 - Dijkstra] Não existe rota de {origem} para {destino}.")
            else:
                caminho = []
                atual = destino
                while atual is not None:
                    caminho.append(atual)
                    atual = predecessores[atual]
                caminho.reverse()
                print("\n[Técnica 1 - Caminho mínimo (Dijkstra)]")
                print("Rota:", " -> ".join(caminho))
                print(f"Custo total da rota: {distancias[destino]}")
        except Exception as e:
            print(f"Erro ao executar Dijkstra: {e}")

        try:
            custo, mst = grafo.prim()
            print("\n[Técnica 2 - Árvore Geradora Mínima (Prim)]")
            if custo == 0 or not mst:
                print("O grafo não é conexo; não foi possível gerar uma AGM única.")
            else:
                print(f"Custo total da AGM: {custo}")
                for u, v, peso in mst:
                    print(f"{u} -> {v} (peso {peso})")
        except Exception as e:
            print(f"Erro ao aplicar Prim: {e}")

        try:
            print("\n[Técnica 3 - Centralidade de Proximidade]")
            cent = grafo.centralidade_proximidade()
            ordenado = sorted(cent.items(), key=lambda x: x[1], reverse=True)
            print("Países mais centrais (hubs comerciais):")
            for nome, valor in ordenado[:3]:
                print(f"{nome}: {valor:.3f}")
        except Exception as e:
            print(f"Erro ao calcular centralidade de proximidade: {e}")

        try:
            print("\n[Técnica 4 - Blocos de comércio (SCC)]")
            comp_id, comps = grafo.scc()
            if not comps:
                print("Nenhuma componente encontrada (grafo vazio?).")
            else:
                for i, comp in enumerate(comps):
                    nomes = [grafo.indices[idx] for idx in comp]
                    print(f"Bloco {i}: {', '.join(nomes)}")
        except Exception as e:
            print(f"Erro ao calcular componentes fortemente conexas: {e}")

    elif numeroMenu == 17:
        if grafo.n == 0:
            print("Grafo vazio. Use a opção 1 para carregar um arquivo primeiro.")
            return

        print("\n=== Descobrindo características do grafo de Rotas Comerciais ===")

        print("\n[Característica 1 - Graus dos vértices]")
        grafo.listarGraus()

        print("\n[Característica 2 - Fontes e Sorvedouros]")
        fontes = []
        sorvedouros = []
        for nome in grafo.nomes:
            try:
                if grafo.isSource(nome):
                    fontes.append(nome)
                if grafo.isSorvedouro(nome):
                    sorvedouros.append(nome)
            except ValueError:
                pass

        print("Países que apenas exportam (fontes):", ", ".join(fontes) if fontes else "nenhum.")
        print("Países que apenas importam (sorvedouros):", ", ".join(sorvedouros) if sorvedouros else "nenhum.")

        print("\n[Característica 3 - Conectividade e Caminho Euleriano]")
        cat = grafo.categoriaConexidade()
        desc = {
            3: "C3 (fortemente conexo)",
            2: "C2 (unilateral)",
            1: "C1 (fracamente conexo)",
            0: "C0 (desconexo)"
        }.get(cat, "Categoria desconhecida")
        print(f"Conectividade do grafo direcionado: {desc}")

        if grafo.hConexidade():
            print("O grafo é h-conexo.")
        else:
            print("O grafo não é h-conexo.")

        if grafo.caminhoEuleriano():
            print("O grafo possui caminho euleriano.")
        else:
            print("O grafo NÃO possui caminho euleriano.")

            
    elif numeroMenu == -1:
        print("Programa encerrado.")
        return

    else:
        print("Opção inválida. Tente novamente.")

def mostrarOpcoes():
    print("""
    Digite 1 para Ler dados do arquivo grafo.txt
    Digite 2 para Gravar dados no arquivo grafo.txt;
    Digite 3 para Inserir vértice;
    Digite 4 para Inserir aresta;
    Digite 5 para Remover vértice;
    Digite 6 para Remover aresta;
    Digite 7 para Mostrar grafo reduzido;
    Digite 8 para Mostrar grafo em lista ou matriz;
    Digite 9 para Apresentar a conexidade do grafo;
    Digite 10 para Verificar se o grafo é h-conexo;
    Digite 11 para Exibir a árvore geradora mínima (Algoritmo de Prim);
    Digite 12 para Ver o grau dos vértices do grafo;
    Digite 13 para saber se o grafo possui caminho Euleriano;
    Digite 14 para visualizar o Grafo plotado;
    Digite 15 para Mostrar o conteúdo do arquivo (legível);
    Digite 16 para Aplicar técnicas ao problema de Rotas Comerciais;
    Digite 17 para Descobrir características do grafo de Rotas Comerciais;
    Digite -1 para Encerrar a aplicação.
    """)

def main():
    numeroMenu = 0
    mostrarTitulo(grafo)
    mostrarOpcoes()
    while numeroMenu != -1:
        numeroMenu = input("Escreva uma das opções (digite 'help' para ver todas as opções): ")
        if numeroMenu == 'help':
            mostrarOpcoes()
        else:
            try:
                numeroMenu = int(numeroMenu)
                switch(numeroMenu)
            except ValueError:
                print("Por favor, insira um número válido.")
                mostrarOpcoes() 

main()

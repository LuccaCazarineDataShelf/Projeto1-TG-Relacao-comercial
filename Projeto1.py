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

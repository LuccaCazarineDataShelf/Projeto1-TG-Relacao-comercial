# Grafos – Projeto P1 (Rotas Comerciais entre Países)

Este repositório contém a implementação de um **grafo direcionado e ponderado** para o problema _“Rotas Comerciais entre Países”_.  
A aplicação em modo texto permite **carregar/salvar** grafos no formato do trabalho, **inserir/remover** vértices/arestas, **calcular conectividade** (C3/C2/C1/C0), **contrair SCCs** (grafo reduzido – DAG), **percursos DFS/BFS**, **Prim**, **Dijkstra**, **grau dos vértices**, **caminho euleriano** e **plotar** o grafo e o grafo reduzido.

---

 ## ✅ Requisitos atendidos (resumo)
    - [x] Leitura/gravação de arquivos `grafo.txt` no **formato do projeto**.
    - [x] **Impressão legível** do arquivo (opção 15).
    - [x] Inserção/remoção de vértices/arestas.
    - [x] **Conexidade para grafos direcionados** com classificação **C3/C2/C1/C0**.
    - [x] **SCC (Kosaraju)** e **grafo reduzido (DAG)**.
    - [x] DFS/BFS; graus, fonte/sumidouro.
    - [x] Prim e Dijkstra (quando aplicável).
    - [x] **Plot** do grafo (PNG) com heurísticas para grafos grandes; **plot do reduzido**.
    - [x] Título coerente exibido acima do menu.

---

## 📦 Dependências
    - Python **3.10+**
    - `networkx`
    - `matplotlib`
    - (opcional) `pygraphviz` + Graphviz para layouts `sfdp/dot`

Instalação rápida:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install networkx matplotlib pygraphviz
    ```

---

## ▶️ Como executar 
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install networkx matplotlib
    python Projeto1.py

---

## 📑 Formato do arquivo
    ```
    <tipo>
    <n>
    <id0> <rótulo com espaços>
    ...
    <m>
    <u> <v> <peso>
    ```
    Tipo `6` = direcionado com pesos nas arestas.

---

## 🧪 Testes rápidos
    Crie arquivos `teste_C3.txt`, `teste_C2.txt`, `teste_C1.txt`, `teste_C0.txt` conforme exemplos e use **1**, **9**, **7**.

---

## 🛠 Problemas comuns
    - Sem GUI: a opção 14 **salva** PNG (warning do backend Agg é normal).
    - `ModuleNotFoundError`: instale dependências.

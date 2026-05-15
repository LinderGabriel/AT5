import csv
import os
import traceback
from pathlib import Path
from collections import Counter, defaultdict
import statistics

PASTA_DO_SCRIPT = Path(__file__).resolve().parent

def pausar():
    print("\n" + "=" * 60)
    os.system("pause")

def encontrar_train_csv():
    caminhos = [
        PASTA_DO_SCRIPT / "train.csv",
        PASTA_DO_SCRIPT / "data" / "train.csv",
        Path.cwd() / "train.csv",
        Path.cwd() / "data" / "train.csv",
    ]

    for caminho in caminhos:
        if caminho.exists():
            return caminho

    raise FileNotFoundError(
        "Arquivo train.csv não encontrado.\n\n"
        "Coloque o arquivo train.csv na MESMA pasta do analise_titanic.py.\n"
        "Ou crie uma pasta chamada data e coloque o train.csv dentro dela."
    )

def carregar_csv(caminho):
    with open(caminho, "r", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        dados = list(leitor)
        colunas = leitor.fieldnames
    return dados, colunas

def converter_numero(valor):
    if valor is None or valor == "":
        return None
    try:
        if "." in valor:
            return float(valor)
        return int(valor)
    except ValueError:
        return None

def detectar_tipo(dados, coluna):
    valores = [linha[coluna] for linha in dados if linha[coluna] != ""]
    if not valores:
        return "texto"

    numericos = [converter_numero(v) for v in valores]
    if all(v is not None for v in numericos):
        if all(float(v).is_integer() for v in numericos):
            return "inteiro"
        return "decimal"

    return "texto"

def resumo_numerico(dados, coluna):
    valores = [converter_numero(linha[coluna]) for linha in dados if converter_numero(linha[coluna]) is not None]

    if not valores:
        return None

    return {
        "quantidade": len(valores),
        "media": statistics.mean(valores),
        "minimo": min(valores),
        "maximo": max(valores),
        "mediana": statistics.median(valores),
    }

def resumo_texto(dados, coluna):
    valores = [linha[coluna] for linha in dados if linha[coluna] != ""]
    contagem = Counter(valores)

    if not valores:
        return None

    mais_frequente, frequencia = contagem.most_common(1)[0]
    return {
        "quantidade": len(valores),
        "valores_unicos": len(contagem),
        "mais_frequente": mais_frequente,
        "frequencia": frequencia,
    }

def grafico_barras(nome_arquivo, titulo, labels, valores):
    maior = max(valores) if valores else 1

    linhas = []
    linhas.append(titulo)
    linhas.append("=" * len(titulo))
    linhas.append("")

    for label, valor in zip(labels, valores):
        tamanho_barra = int((valor / maior) * 40)
        barra = "#" * tamanho_barra
        linhas.append(f"{label}: {barra} ({valor})")

    with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas))

def grafico_histograma_idades(nome_arquivo, idades):
    faixas = {
        "0-10": 0,
        "11-20": 0,
        "21-30": 0,
        "31-40": 0,
        "41-50": 0,
        "51-60": 0,
        "61-70": 0,
        "71-80": 0,
    }

    for idade in idades:
        if idade <= 10:
            faixas["0-10"] += 1
        elif idade <= 20:
            faixas["11-20"] += 1
        elif idade <= 30:
            faixas["21-30"] += 1
        elif idade <= 40:
            faixas["31-40"] += 1
        elif idade <= 50:
            faixas["41-50"] += 1
        elif idade <= 60:
            faixas["51-60"] += 1
        elif idade <= 70:
            faixas["61-70"] += 1
        else:
            faixas["71-80"] += 1

    grafico_barras(
        nome_arquivo,
        "Distribuicao das idades dos passageiros",
        list(faixas.keys()),
        list(faixas.values())
    )

try:
    print("Iniciando a atividade Titanic...\n")

    pasta_graficos = PASTA_DO_SCRIPT / "graficos"
    pasta_graficos.mkdir(exist_ok=True)

    caminho_csv = encontrar_train_csv()
    print(f"Arquivo encontrado: {caminho_csv}\n")

    dados, colunas = carregar_csv(caminho_csv)

    print("1) QUANTIDADE DE LINHAS E COLUNAS")
    print(f"Linhas: {len(dados)}")
    print(f"Colunas: {len(colunas)}")

    print("\n2) TIPO DE DADO DE CADA VARIAVEL")
    tipos = {}
    for coluna in colunas:
        tipo = detectar_tipo(dados, coluna)
        tipos[coluna] = tipo
        print(f"{coluna}: {tipo}")

    print("\n3) RESUMO ESTATISTICO DAS VARIAVEIS")
    for coluna in colunas:
        print(f"\nVariavel: {coluna}")
        if tipos[coluna] in ["inteiro", "decimal"]:
            resumo = resumo_numerico(dados, coluna)
            if resumo:
                print(f"Quantidade: {resumo['quantidade']}")
                print(f"Media: {resumo['media']:.2f}")
                print(f"Minimo: {resumo['minimo']}")
                print(f"Maximo: {resumo['maximo']}")
                print(f"Mediana: {resumo['mediana']}")
        else:
            resumo = resumo_texto(dados, coluna)
            if resumo:
                print(f"Quantidade: {resumo['quantidade']}")
                print(f"Valores unicos: {resumo['valores_unicos']}")
                print(f"Mais frequente: {resumo['mais_frequente']}")
                print(f"Frequencia: {resumo['frequencia']}")

    print("\n4) TABELA DE FREQUENCIA DA VARIAVEL SURVIVED")
    survived = Counter(linha["Survived"] for linha in dados)
    for categoria in sorted(survived):
        print(f"{categoria}: {survived[categoria]}")

    grafico_barras(
        pasta_graficos / "survival.txt",
        "Grafico de barras - Survival",
        ["0 - Nao sobreviveu", "1 - Sobreviveu"],
        [survived["0"], survived["1"]]
    )

    print("\nO grafico de barras foi escolhido porque a variavel Survived e categorica.")
    print("Ele facilita a comparacao entre quem sobreviveu e quem nao sobreviveu.")

    print("\n5) QUANTAS PESSOAS SOBREVIVERAM? QUAL A CLASSE MAIS FREQUENTE?")
    print(f"Pessoas que sobreviveram: {survived['1']}")

    classes = Counter(linha["Pclass"] for linha in dados)
    classe_mais_frequente = classes.most_common(1)[0][0]
    print(f"Classe mais frequente: {classe_mais_frequente}")

    print("\n6) RELACAO ENTRE SEX E SURVIVED")
    tabela_sex_survival = defaultdict(lambda: Counter())

    for linha in dados:
        tabela_sex_survival[linha["Sex"]][linha["Survived"]] += 1

    print("Sexo       Nao sobreviveu   Sobreviveu")
    for sexo in sorted(tabela_sex_survival):
        print(f"{sexo:<10} {tabela_sex_survival[sexo]['0']:<15} {tabela_sex_survival[sexo]['1']}")

    grafico_barras(
        pasta_graficos / "sex_survival.txt",
        "Grafico de barras - Sexo x Survival",
        ["Female nao sobreviveu", "Female sobreviveu", "Male nao sobreviveu", "Male sobreviveu"],
        [
            tabela_sex_survival["female"]["0"],
            tabela_sex_survival["female"]["1"],
            tabela_sex_survival["male"]["0"],
            tabela_sex_survival["male"]["1"],
        ]
    )

    mulheres_sobreviveram = tabela_sex_survival["female"]["1"]
    homens_sobreviveram = tabela_sex_survival["male"]["1"]

    print(f"\nMulheres que sobreviveram: {mulheres_sobreviveram}")
    print(f"Homens que sobreviveram: {homens_sobreviveram}")
    print("Sim, mulheres sobreviveram mais que homens.")

    print("\n7) DISTRIBUICAO DAS IDADES DOS PASSAGEIROS")
    idades = [float(linha["Age"]) for linha in dados if linha["Age"] != ""]

    grafico_histograma_idades(pasta_graficos / "distribuicao_idades.txt", idades)

    print(f"Media das idades: {statistics.mean(idades):.2f}")
    print(f"Mediana das idades: {statistics.median(idades):.2f}")
    print("A distribuicao nao parece perfeitamente simetrica.")
    print("A maioria dos passageiros era jovem/adulta, nao idosa.")

    print("\nAtividade finalizada com sucesso.")
    print(f"Os graficos foram salvos em: {pasta_graficos}")

except Exception:
    print("\nOCORREU UM ERRO:")
    print(traceback.format_exc())

finally:
    pausar()

"""Preparação, divisão e métricas do estudo de fraude.

O módulo não treina os modelos. Ele concentra as regras que os testes
precisam travar: remoção de duplicatas antes do split, ausência de
vazamento de transação exata, log1p e padronização ajustados só no
treino, peso da classe positiva só no treino, e limiar escolhido por F2.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.metrics import auc as area_trapezoidal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

URL_CSV_PUBLICO = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
URL_TUTORIAL_TENSORFLOW = "https://www.tensorflow.org/tutorials/structured_data/imbalanced_data"
URL_DATASET_KAGGLE = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
URL_LICENCA_PAGINA_KAGGLE = "http://opendatacommons.org/licenses/dbcl/1.0/"

SHA256_ARQUIVO_PUBLICO = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
LINHAS_ARQUIVO_PUBLICO = 284_807
NORMAIS_ARQUIVO_PUBLICO = 284_315
FRAUDES_ARQUIVO_PUBLICO = 492

SEMENTE_PADRAO = 42
FRACAO_TESTE = 0.20
FRACAO_VALIDACAO_DO_RESTANTE = 0.25
BETA_F = 2.0

COLUNAS_V = [f"V{i}" for i in range(1, 29)]
COLUNAS_BRUTAS = ["Time", *COLUNAS_V, "Amount"]
COLUNA_ALVO = "Class"
NOMES_MODELO = ["Time_z", *[f"V{i}_z" for i in range(1, 29)], "Amount_log1p_z"]


def sha256_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with Path(caminho).open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            digest.update(bloco)
    return digest.hexdigest()


def fingerprint_tabela(df: pd.DataFrame) -> str:
    """Hash do conteúdo tabular, distinto do hash do arquivo bruto."""
    texto = df.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def conferir_arquivo_bruto(caminho: Path) -> dict:
    caminho = Path(caminho)
    digest = sha256_arquivo(caminho)
    if digest != SHA256_ARQUIVO_PUBLICO:
        raise ValueError(
            f"sha256 inesperado para {caminho}: {digest} != {SHA256_ARQUIVO_PUBLICO}"
        )
    df = pd.read_csv(caminho)
    contagem = df[COLUNA_ALVO].value_counts().to_dict()
    relatorio = {
        "caminho": str(caminho),
        "sha256": digest,
        "bytes": caminho.stat().st_size,
        "linhas": int(len(df)),
        "colunas": int(df.shape[1]),
        "nomes": list(df.columns),
        "nulos": int(df.isna().sum().sum()),
        "normais": int(contagem.get(0, 0)),
        "fraudes": int(contagem.get(1, 0)),
    }
    if relatorio["linhas"] != LINHAS_ARQUIVO_PUBLICO:
        raise ValueError(f"contagem de linhas inesperada: {relatorio['linhas']}")
    if relatorio["nomes"] != [*COLUNAS_BRUTAS, COLUNA_ALVO]:
        raise ValueError(f"esquema inesperado: {relatorio['nomes']}")
    if relatorio["normais"] != NORMAIS_ARQUIVO_PUBLICO or relatorio["fraudes"] != FRAUDES_ARQUIVO_PUBLICO:
        raise ValueError(f"contagem de classes inesperada: {contagem}")
    if relatorio["nulos"] != 0:
        raise ValueError(f"o arquivo público não deveria ter nulos: {relatorio['nulos']}")
    return relatorio


def _exigir_esquema(df: pd.DataFrame) -> None:
    faltantes = [coluna for coluna in [*COLUNAS_BRUTAS, COLUNA_ALVO] if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"colunas ausentes: {faltantes}")


def contar_conflitos_de_classe(df: pd.DataFrame) -> int:
    """Grupos com os mesmos atributos e rótulos diferentes."""
    _exigir_esquema(df)
    mascara = df.duplicated(subset=COLUNAS_BRUTAS, keep=False)
    if not bool(mascara.any()):
        return 0
    grupos = df.loc[mascara].groupby(COLUNAS_BRUTAS, sort=False)[COLUNA_ALVO].nunique()
    return int((grupos > 1).sum())


def remover_duplicatas_exatas(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove cópias exatas de linha inteira antes de qualquer split.

    A primeira ocorrência no arquivo é mantida. Cópias extras da mesma
    transação, se ficassem livres para cair em treino e teste, repetiriam
    o mesmo exemplo dos dois lados e inflariam a métrica.
    """
    _exigir_esquema(df)
    extras = int(df.duplicated().sum())
    envolvidas = int(df.duplicated(keep=False).sum())
    conflitos = contar_conflitos_de_classe(df)
    duplicatas_de_atributos = int(df.duplicated(subset=COLUNAS_BRUTAS).sum())
    fraude_antes = int((df[COLUNA_ALVO] == 1).sum())
    normal_antes = int((df[COLUNA_ALVO] == 0).sum())
    limpo = df.drop_duplicates(keep="first").reset_index(drop=True)
    relatorio = {
        "linhas_brutas": int(len(df)),
        "duplicatas_exatas_extras": extras,
        "linhas_em_grupos_duplicados": envolvidas,
        "grupos_duplicados": int(envolvidas - extras),
        "duplicatas_de_atributos_extras_antes": duplicatas_de_atributos,
        "conflitos_de_classe": conflitos,
        "fraude_antes": fraude_antes,
        "normal_antes": normal_antes,
        "linhas_apos_remocao": int(len(limpo)),
        "fraude_depois": int((limpo[COLUNA_ALVO] == 1).sum()),
        "normal_depois": int((limpo[COLUNA_ALVO] == 0).sum()),
        "duplicatas_exatas_extras_depois": int(limpo.duplicated().sum()),
        "decisao": "remover_duplicata_exata_antes_do_split",
        "criterio_keep": "first",
    }
    return limpo, relatorio


def dividir_estratificado(
    df: pd.DataFrame,
    semente: int = SEMENTE_PADRAO,
    fracao_teste: float = FRACAO_TESTE,
    fracao_validacao_do_restante: float = FRACAO_VALIDACAO_DO_RESTANTE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """60/20/20 por padrão, estratificado pela classe, sem reordenar o alvo."""
    _exigir_esquema(df)
    if df[COLUNA_ALVO].nunique() < 2:
        raise ValueError("o split estratificado precisa das duas classes")
    treino_validacao, teste = train_test_split(
        df,
        test_size=fracao_teste,
        random_state=semente,
        stratify=df[COLUNA_ALVO],
    )
    treino, validacao = train_test_split(
        treino_validacao,
        test_size=fracao_validacao_do_restante,
        random_state=semente,
        stratify=treino_validacao[COLUNA_ALVO],
    )
    return (
        treino.reset_index(drop=True),
        validacao.reset_index(drop=True),
        teste.reset_index(drop=True),
    )


def preparar_particoes(
    df: pd.DataFrame,
    semente: int = SEMENTE_PADRAO,
    fracao_teste: float = FRACAO_TESTE,
    fracao_validacao_do_restante: float = FRACAO_VALIDACAO_DO_RESTANTE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Limpa duplicatas e só então sorteia treino, validação e teste.

    Se sobrar um grupo com os mesmos atributos e rótulos conflitantes,
    o grupo inteiro vai para o treino. Ele não entra na validação nem no
    teste, então a contradição não vaza para a métrica. No arquivo público
    conferido neste estudo esse resto é zero; a regra existe para o caso
    geral e é coberta por teste.
    """
    limpo, relatorio = remover_duplicatas_exatas(df)
    mascara = limpo.duplicated(subset=COLUNAS_BRUTAS, keep=False)
    grupos = limpo.loc[mascara].copy()
    resto = limpo.loc[~mascara].copy()
    treino, validacao, teste = dividir_estratificado(
        resto,
        semente=semente,
        fracao_teste=fracao_teste,
        fracao_validacao_do_restante=fracao_validacao_do_restante,
    )
    if len(grupos):
        treino = pd.concat([treino, grupos], ignore_index=True)
    relatorio["linhas_de_grupos_atributo_forcadas_no_treino"] = int(len(grupos))
    relatorio["linhas_elegiveis_ao_split"] = int(len(resto))
    relatorio["semente"] = int(semente)
    relatorio["fracao_teste"] = float(fracao_teste)
    relatorio["fracao_validacao_do_restante"] = float(fracao_validacao_do_restante)
    relatorio["treino"] = int(len(treino))
    relatorio["validacao"] = int(len(validacao))
    relatorio["teste"] = int(len(teste))
    relatorio["sobreposicao_atributos"] = sobreposicao_atributos(treino, validacao, teste)
    return treino, validacao, teste, relatorio


def _chaves_atributos(df: pd.DataFrame) -> np.ndarray:
    """Chave binária dos atributos. Igualdade exata de float64, sem colisão de hash."""
    if df.empty:
        return np.empty(0, dtype=np.dtype((np.void, 8)))
    matriz = np.ascontiguousarray(df.loc[:, COLUNAS_BRUTAS].to_numpy(dtype=np.float64))
    tipo = np.dtype((np.void, matriz.shape[1] * matriz.dtype.itemsize))
    return matriz.view(tipo).ravel()


def sobreposicao_atributos(
    treino: pd.DataFrame,
    validacao: pd.DataFrame,
    teste: pd.DataFrame,
) -> dict:
    """Conta vetores de atributos distintos compartilhados entre partições."""

    def pares(esquerda: pd.DataFrame, direita: pd.DataFrame) -> int:
        if esquerda.empty or direita.empty:
            return 0
        comum = np.intersect1d(_chaves_atributos(esquerda), _chaves_atributos(direita))
        return int(comum.size)

    return {
        "treino_validacao": pares(treino, validacao),
        "treino_teste": pares(treino, teste),
        "validacao_teste": pares(validacao, teste),
    }


def matriz_pre_escala(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica log1p em Amount. A transformação não estima parâmetro."""
    _exigir_esquema(df)
    if bool((df["Amount"] < 0).any()):
        raise ValueError("Amount negativo impediria um log1p interpretável neste estudo")
    atributos = df.loc[:, COLUNAS_BRUTAS].copy()
    atributos["Amount"] = np.log1p(atributos["Amount"].to_numpy(dtype=np.float64))
    return atributos.rename(columns={"Amount": "Amount_log1p"})


def ajustar_atributos(
    treino: pd.DataFrame,
    *particoes: pd.DataFrame,
) -> tuple[StandardScaler, pd.DataFrame, list[pd.DataFrame]]:
    """Ajusta o StandardScaler exclusivamente no treino e aplica nas demais."""
    base = matriz_pre_escala(treino)
    scaler = StandardScaler()
    treino_z = pd.DataFrame(
        scaler.fit_transform(base.to_numpy(dtype=np.float64)),
        columns=NOMES_MODELO,
    )
    transformadas = []
    for particao in particoes:
        matriz = matriz_pre_escala(particao)
        transformadas.append(
            pd.DataFrame(
                scaler.transform(matriz.to_numpy(dtype=np.float64)),
                columns=NOMES_MODELO,
            )
        )
    return scaler, treino_z, transformadas


def scale_pos_weight_treino(y_treino: np.ndarray | pd.Series) -> float:
    """Razão negativos/positivos medida só no treino, para o XGBoost."""
    y = np.asarray(y_treino)
    positivos = int(np.sum(y == 1))
    negativos = int(np.sum(y == 0))
    if positivos == 0:
        raise ValueError("treino sem fraude: scale_pos_weight indefinido")
    return float(negativos / positivos)


def fbeta_de_contagens(tn: int, fp: int, fn: int, tp: int, beta: float = BETA_F) -> dict:
    precisao = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    beta2 = float(beta) ** 2
    denominador = beta2 * precisao + recall
    fbeta = (1.0 + beta2) * precisao * recall / denominador if denominador else 0.0
    denominador_f1 = precisao + recall
    f1 = 2.0 * precisao * recall / denominador_f1 if denominador_f1 else 0.0
    return {
        "precisao": float(precisao),
        "recall": float(recall),
        "f1": float(f1),
        "fbeta": float(fbeta),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def varrer_limiares(y_true, scores, beta: float = BETA_F) -> dict:
    """Avalia cada limiar realizável pela regra escore >= limiar.

    Linhas com o mesmo escore entram juntas. Um limiar que classificasse
    só parte de um empate não é uma regra do classificador.
    """
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y e escores precisam ser vetores do mesmo tamanho")
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("o cálculo do limiar precisa das duas classes")
    ordem = np.argsort(-s, kind="mergesort")
    y_ord = y[ordem]
    s_ord = s[ordem]
    tp = np.cumsum(y_ord == 1, dtype=np.int64)
    fp = np.cumsum(y_ord == 0, dtype=np.int64)
    if len(s_ord) == 1:
        fins = np.array([0], dtype=np.int64)
    else:
        mudancas = np.flatnonzero(s_ord[1:] != s_ord[:-1])
        fins = np.concatenate([mudancas, np.array([len(s_ord) - 1], dtype=np.int64)])
    tp_e = tp[fins]
    fp_e = fp[fins]
    fn_e = n_pos - tp_e
    tn_e = n_neg - fp_e
    precisao = np.divide(
        tp_e,
        tp_e + fp_e,
        out=np.zeros(len(fins), dtype=np.float64),
        where=(tp_e + fp_e) > 0,
    )
    recall = tp_e / n_pos
    beta2 = float(beta) ** 2
    denominador = beta2 * precisao + recall
    fbeta = np.divide(
        (1.0 + beta2) * precisao * recall,
        denominador,
        out=np.zeros(len(fins), dtype=np.float64),
        where=denominador > 0,
    )
    denominador_f1 = precisao + recall
    f1 = np.divide(
        2.0 * precisao * recall,
        denominador_f1,
        out=np.zeros(len(fins), dtype=np.float64),
        where=denominador_f1 > 0,
    )
    return {
        "limiar": s_ord[fins],
        "precisao": precisao,
        "recall": recall,
        "f1": f1,
        "fbeta": fbeta,
        "tp": tp_e,
        "fp": fp_e,
        "fn": fn_e,
        "tn": tn_e,
    }


def escolher_limiar_fbeta(y_true, scores, beta: float = BETA_F) -> dict:
    """Maximiza o F-beta. No empate, fica o maior limiar, com menos alertas.

    A função só recebe os rótulos e os escores da partição de decisão.
    Quem chama é responsável por passar a validação, não o teste.
    """
    curva = varrer_limiares(y_true, scores, beta=beta)
    melhor = float(np.max(curva["fbeta"]))
    candidatos = np.flatnonzero(np.abs(curva["fbeta"] - melhor) <= 1e-12)
    posicao_local = int(np.argmax(curva["limiar"][candidatos]))
    j = int(candidatos[posicao_local])
    return {
        "limiar": float(curva["limiar"][j]),
        "fbeta": float(curva["fbeta"][j]),
        "precisao": float(curva["precisao"][j]),
        "recall": float(curva["recall"][j]),
        "f1": float(curva["f1"][j]),
        "tp": int(curva["tp"][j]),
        "fp": int(curva["fp"][j]),
        "fn": int(curva["fn"][j]),
        "tn": int(curva["tn"][j]),
        "beta": float(beta),
        "regra": "maximo_fbeta_empate_no_maior_limiar",
    }


def metricas_operacao(y_true, scores, limiar: float, beta: float = BETA_F) -> dict:
    """Métricas da classe positiva e custo de alerta num limiar já decidido."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y e escores precisam ser vetores do mesmo tamanho")
    pred = (s >= float(limiar)).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    taxas = fbeta_de_contagens(int(tn), int(fp), int(fn), int(tp), beta=beta)
    n = int(len(y))
    alertas = int(tp + fp)
    precisao_curva, recall_curva, _ = precision_recall_curve(y, s)
    ap = float(average_precision_score(y, s))
    roc = float(roc_auc_score(y, s))
    trapezio = float(area_trapezoidal(recall_curva, precisao_curva))
    return {
        "limiar": float(limiar),
        "beta": float(beta),
        "precisao_fraude": taxas["precisao"],
        "recall_fraude": taxas["recall"],
        "f1_fraude": taxas["f1"],
        "f2_fraude": taxas["fbeta"],
        "average_precision": ap,
        "pr_auc_trapezio": trapezio,
        "roc_auc": roc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "alertas": alertas,
        "alertas_por_mil": float(1000.0 * alertas / n),
        "falsos_positivos_por_mil": float(1000.0 * fp / n),
        "n": n,
        "prevalencia": float(np.mean(y)),
        "acuracia": float((tp + tn) / n),
    }


def amostra_explicacao(y_true, n_normais: int = 1500, semente: int = SEMENTE_PADRAO) -> np.ndarray:
    """Índices da validação para o SHAP global.

    Entram todas as fraudes da partição e uma amostra reprodutível de
    legítimas. A fraude fica sobrerrepresentada de propósito: uma amostra
    proporcional quase não mostraria a classe rara. Isso não retreina o
    modelo nem escolhe limiar.
    """
    y = np.asarray(y_true)
    positivos = np.flatnonzero(y == 1)
    negativos = np.flatnonzero(y == 0)
    if len(positivos) == 0 or len(negativos) == 0:
        raise ValueError("a amostra de explicação precisa das duas classes")
    gerador = np.random.default_rng(semente)
    quantidade = min(int(n_normais), len(negativos))
    escolhidos = gerador.choice(negativos, size=quantidade, replace=False)
    return np.sort(np.concatenate([positivos, escolhidos]))


def indice_representativo(y_true, scores, limiar: float, tipo: str):
    """Caso mais próximo da mediana do grupo, com desempate no menor índice.

    tipo é tp, fp ou fn. A escolha é ilustrativa e deve ocorrer depois da
    métrica final. Não devolve um caso se o grupo estiver vazio.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=np.float64)
    predito = s >= float(limiar)
    if tipo == "tp":
        mascara = (y == 1) & predito
    elif tipo == "fp":
        mascara = (y == 0) & predito
    elif tipo == "fn":
        mascara = (y == 1) & ~predito
    else:
        raise ValueError("tipo deve ser tp, fp ou fn")
    indices = np.flatnonzero(mascara)
    if len(indices) == 0:
        return None
    alvo = float(np.median(s[indices]))
    distancia = np.abs(s[indices] - alvo)
    melhor = float(distancia.min())
    candidatos = indices[np.abs(distancia - melhor) <= 1e-12]
    return int(candidatos.min())


def contagens_classe(df: pd.DataFrame) -> dict:
    contagem = df[COLUNA_ALVO].value_counts().to_dict()
    n = int(len(df))
    fraudes = int(contagem.get(1, 0))
    return {
        "n": n,
        "normais": int(contagem.get(0, 0)),
        "fraudes": fraudes,
        "prevalencia": float(fraudes / n) if n else 0.0,
    }

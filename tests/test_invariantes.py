"""Invariantes de split, transformação e limiar. Não usam o CSV de 150 MB."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from src.fraude_cartao import (
    COLUNAS_BRUTAS,
    FRACAO_TESTE,
    NOMES_MODELO,
    SHA256_ARQUIVO_PUBLICO,
    ajustar_atributos,
    amostra_explicacao,
    escolher_limiar_fbeta,
    fbeta_de_contagens,
    fingerprint_tabela,
    indice_representativo,
    matriz_pre_escala,
    metricas_operacao,
    preparar_particoes,
    remover_duplicatas_exatas,
    scale_pos_weight_treino,
    sha256_arquivo,
    sobreposicao_atributos,
    varrer_limiares,
)

ROOT = Path(__file__).resolve().parents[1]


def quadro(n=80, semente=0, fraude=8):
    rng = np.random.default_rng(semente)
    dados = {
        "Time": np.arange(n, dtype=float) * 10,
        "Amount": rng.uniform(0, 400, n),
        "Class": np.zeros(n, dtype=int),
    }
    dados["Class"][:fraude] = 1
    rng.shuffle(dados["Class"])
    for indice in range(1, 29):
        dados[f"V{indice}"] = rng.normal(size=n)
    return pd.DataFrame(dados)


def test_f2_tem_a_formula_e_concorda_com_o_sklearn():
    taxas = fbeta_de_contagens(tn=90, fp=10, fn=2, tp=8, beta=2)
    assert taxas["precisao"] == pytest.approx(8 / 18)
    assert taxas["recall"] == pytest.approx(8 / 10)
    esperado = (5 * taxas["precisao"] * taxas["recall"]) / (4 * taxas["precisao"] + taxas["recall"])
    assert taxas["fbeta"] == pytest.approx(esperado)
    y = np.array([0] * 100 + [1] * 10)
    pred = np.array([0] * 90 + [1] * 10 + [0] * 2 + [1] * 8)
    assert taxas["fbeta"] == pytest.approx(fbeta_score(y, pred, beta=2, zero_division=0))
    assert taxas["precisao"] == pytest.approx(precision_score(y, pred, zero_division=0))
    assert taxas["recall"] == pytest.approx(recall_score(y, pred, zero_division=0))


def test_limiar_f2_nao_escolhe_classificar_tudo_como_fraude():
    y = np.array([0] * 20 + [1] * 5)
    scores = np.array([0.1] * 20 + [0.9] * 5)
    escolha = escolher_limiar_fbeta(y, scores, beta=2)
    assert escolha["limiar"] == pytest.approx(0.9)
    assert escolha["recall"] == pytest.approx(1)
    assert escolha["precisao"] == pytest.approx(1)
    assert escolha["fbeta"] == pytest.approx(1)
    todos = metricas_operacao(y, scores, limiar=0.1, beta=2)
    assert todos["recall_fraude"] == pytest.approx(1)
    assert todos["precisao_fraude"] == pytest.approx(0.2)
    assert escolha["fbeta"] > todos["f2_fraude"]


def test_limiar_meio_nao_e_assumido_e_empate_sobe_o_corte():
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.1, 0.2, 0.2, 0.4, 0.4])
    escolha = escolher_limiar_fbeta(y, scores, beta=2)
    referencia = metricas_operacao(y, scores, limiar=0.5, beta=2)
    assert referencia["recall_fraude"] == 0
    assert escolha["fbeta"] > referencia["f2_fraude"]
    assert escolha["limiar"] == pytest.approx(0.4)

    # Dois cortes com o mesmo F2: 0.8 pega os dois fraudes; 0.95 pega um.
    # Precisão e recall diferentes, então montamos um empate artificial na varredura
    # por escores repetidos que produzem o mesmo ponto. O desempate é coberto
    # quando vários escores positivos idênticos compartilham o máximo.
    y2 = np.array([0, 0, 1, 1])
    scores2 = np.array([0.2, 0.3, 0.8, 0.8])
    escolha2 = escolher_limiar_fbeta(y2, scores2, beta=2)
    curva = varrer_limiares(y2, scores2, beta=2)
    maximos = curva["limiar"][np.abs(curva["fbeta"] - curva["fbeta"].max()) <= 1e-12]
    assert escolha2["limiar"] == pytest.approx(float(np.max(maximos)))


def test_varredura_concorda_com_a_regra_maior_ou_igual():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 200)
    y[0] = 0
    y[1] = 1
    scores = rng.random(200)
    scores[10] = scores[11] = 0.5
    escolha = escolher_limiar_fbeta(y, scores, beta=2)
    metricas = metricas_operacao(y, scores, escolha["limiar"], beta=2)
    assert metricas["tp"] == escolha["tp"]
    assert metricas["fp"] == escolha["fp"]
    assert metricas["fn"] == escolha["fn"]
    assert metricas["tn"] == escolha["tn"]
    assert metricas["f2_fraude"] == pytest.approx(escolha["fbeta"])


def test_average_precision_e_trapezio_podem_divergir():
    y = np.array([0, 0, 0, 0, 1, 0, 1, 0, 0, 1])
    scores = np.array([0.05, 0.2, 0.9, 0.3, 0.4, 0.35, 0.8, 0.1, 0.6, 0.15])
    metricas = metricas_operacao(y, scores, limiar=0.5)
    assert metricas["average_precision"] != pytest.approx(metricas["pr_auc_trapezio"])
    assert 0 < metricas["average_precision"] <= 1
    assert 0 <= metricas["pr_auc_trapezio"] <= 1


def test_duplicata_exata_sai_antes_do_split_e_nao_vaza():
    base = quadro(60, semente=1, fraude=6)
    copia = base.iloc[[0, 1, 2, 3, 4]].copy()
    sujo = pd.concat([base, copia], ignore_index=True)
    limpo, relatorio = remover_duplicatas_exatas(sujo)
    assert relatorio["duplicatas_exatas_extras"] == 5
    assert relatorio["linhas_apos_remocao"] == len(base)
    assert relatorio["duplicatas_exatas_extras_depois"] == 0
    assert relatorio["decisao"] == "remover_duplicata_exata_antes_do_split"

    treino, validacao, teste, particao = preparar_particoes(sujo, semente=42)
    assert particao["sobreposicao_atributos"] == {
        "treino_validacao": 0,
        "treino_teste": 0,
        "validacao_teste": 0,
    }
    uniao = len(treino) + len(validacao) + len(teste)
    assert uniao == len(base)
    for parte in (treino, validacao, teste):
        assert set(parte["Class"]).issubset({0, 1})
    assert treino["Class"].sum() > 0
    assert validacao["Class"].sum() > 0
    assert teste["Class"].sum() > 0


def test_conflito_de_rotulo_fica_inteiro_no_treino():
    base = quadro(80, semente=2, fraude=8)
    conflito = base.iloc[[0]].copy()
    conflito["Class"] = 1 - int(conflito["Class"].iloc[0])
    sujo = pd.concat([base, conflito], ignore_index=True)
    treino, validacao, teste, relatorio = preparar_particoes(sujo, semente=3)
    assert relatorio["conflitos_de_classe"] == 1
    assert relatorio["linhas_de_grupos_atributo_forcadas_no_treino"] == 2
    chave = tuple(conflito.iloc[0][COLUNAS_BRUTAS].tolist())

    def chaves(df):
        return {tuple(linha) for linha in df[COLUNAS_BRUTAS].to_numpy().tolist()}

    assert chave in chaves(treino)
    assert chave not in chaves(validacao)
    assert chave not in chaves(teste)
    assert sobreposicao_atributos(treino, validacao, teste)["treino_teste"] == 0
    assert sobreposicao_atributos(treino, validacao, teste)["validacao_teste"] == 0


def test_split_reproduz_e_respeita_a_proporcao():
    df = quadro(200, semente=4, fraude=20)
    a_tr, a_va, a_te, _ = preparar_particoes(df, semente=11)
    b_tr, b_va, b_te, _ = preparar_particoes(df, semente=11)
    pd.testing.assert_frame_equal(a_tr, b_tr)
    pd.testing.assert_frame_equal(a_va, b_va)
    pd.testing.assert_frame_equal(a_te, b_te)
    n = len(df)
    assert len(a_te) / n == pytest.approx(FRACAO_TESTE, abs=0.02)
    assert len(a_va) / n == pytest.approx(0.20, abs=0.02)
    assert len(a_tr) / n == pytest.approx(0.60, abs=0.02)
    prevalencia = df["Class"].mean()
    for parte in (a_tr, a_va, a_te):
        assert parte["Class"].mean() == pytest.approx(prevalencia, abs=0.03)


def test_scaler_ve_so_o_treino_e_o_amount_entra_em_log1p():
    df = quadro(120, semente=5, fraude=12)
    treino, validacao, teste, _ = preparar_particoes(df, semente=5)
    validacao = validacao.copy()
    teste = teste.copy()
    validacao["Amount"] = validacao["Amount"] + 10_000
    teste["Amount"] = 0
    scaler, treino_z, (validacao_z, teste_z) = ajustar_atributos(treino, validacao, teste)

    assert scaler.n_samples_seen_ == len(treino)
    assert list(treino_z.columns) == NOMES_MODELO
    assert "Amount_log1p_z" in treino_z.columns
    assert "Amount" not in treino_z.columns
    np.testing.assert_allclose(treino_z.mean().to_numpy(), 0, atol=1e-8)
    np.testing.assert_allclose(treino_z.std(ddof=0).to_numpy(), 1, atol=1e-8)

    referencia = StandardScaler().fit(matriz_pre_escala(treino).to_numpy())
    np.testing.assert_allclose(scaler.mean_, referencia.mean_)
    vazado = StandardScaler().fit(
        matriz_pre_escala(pd.concat([treino, validacao], ignore_index=True)).to_numpy()
    )
    assert not np.allclose(scaler.mean_, vazado.mean_)

    log_amount = np.log1p(treino["Amount"].to_numpy(dtype=np.float64))
    assert scaler.mean_[-1] == pytest.approx(log_amount.mean())
    assert scaler.mean_[-1] != pytest.approx(treino["Amount"].mean())
    # Validação deslocada não é reescalada para média zero.
    assert abs(float(validacao_z["Amount_log1p_z"].mean())) > 0.5
    np.testing.assert_allclose(
        matriz_pre_escala(treino)["Amount_log1p"].to_numpy(),
        log_amount,
    )
    assert len(teste_z) == len(teste)


def test_scale_pos_weight_ignora_validacao_e_teste():
    y_treino = np.array([0, 0, 0, 0, 1])
    y_validacao = np.array([1, 1, 1, 0])
    peso = scale_pos_weight_treino(y_treino)
    assert peso == pytest.approx(4)
    assert peso != pytest.approx(scale_pos_weight_treino(np.concatenate([y_treino, y_validacao])))
    with pytest.raises(ValueError):
        scale_pos_weight_treino(np.zeros(4, dtype=int))


def test_fingerprint_muda_quando_a_linha_sai_e_o_cache_publico_confere():
    df = quadro(12, semente=6, fraude=2)
    outro = df.drop(index=[0]).reset_index(drop=True)
    assert fingerprint_tabela(df) != fingerprint_tabela(outro)
    caminho = ROOT / "data" / "creditcard.csv"
    if not caminho.exists():
        pytest.skip("cache local do CSV público ausente")
    assert sha256_arquivo(caminho) == SHA256_ARQUIVO_PUBLICO


def test_amostra_e_caso_local_sao_reproduziveis_e_nao_vazam_o_criterio():
    y = np.array([0] * 30 + [1] * 4)
    primeira = amostra_explicacao(y, n_normais=10, semente=5)
    segunda = amostra_explicacao(y, n_normais=10, semente=5)
    np.testing.assert_array_equal(primeira, segunda)
    assert set(np.flatnonzero(y == 1)).issubset(set(primeira))
    assert len(primeira) == 14
    assert len(set(primeira)) == len(primeira)

    scores = np.array([0.1] * 30 + [0.2, 0.4, 0.8, 0.95])
    tp = indice_representativo(y, scores, limiar=0.5, tipo="tp")
    fp = indice_representativo(y, scores, limiar=0.05, tipo="fp")
    fn = indice_representativo(y, scores, limiar=0.99, tipo="fn")
    assert tp is not None and y[tp] == 1 and scores[tp] >= 0.5
    assert fp is not None and y[fp] == 0 and scores[fp] >= 0.05
    assert fn is not None and y[fn] == 1 and scores[fn] < 0.99
    assert indice_representativo(y, scores, limiar=0.99, tipo="tp") is None
    y2 = np.array([1, 1])
    s2 = np.array([0.4, 0.8])
    assert indice_representativo(y2, s2, limiar=0.3, tipo="tp") == 0


def test_sem_alertas_o_f2_e_zero_e_nao_vira_recall_perfeito():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.45])
    metricas = metricas_operacao(y, scores, limiar=0.99)
    assert metricas["tp"] == 0
    assert metricas["fp"] == 0
    assert metricas["recall_fraude"] == 0
    assert metricas["precisao_fraude"] == 0
    assert metricas["f2_fraude"] == 0
    assert metricas["alertas"] == 0

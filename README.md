# Detecção de fraude em cartão de crédito

Estudo executado em português para reconhecer a transação fraudulenta rara. No arquivo público há 284.807 transações e 492 fraudes (0,172749% das linhas). Um classificador que responde sempre "legítima" acerta 99,8326% do teste e não encontra nenhuma fraude. A acurácia, sozinha, não mede o problema.

O notebook com as saídas desta execução está em `notebooks/deteccao_fraude_cartao.ipynb`. As figuras estão em `artifacts/figuras/` e o JSON com as métricas, em `artifacts/metricas.json`. O CSV não entra no git.

## O problema e o desbalanceamento

O arquivo reúne transações de cartão de setembro de 2013, dois dias, portadores europeus, publicadas pelo Machine Learning Group da ULB com a Worldline. Há 31 colunas: `Time`, `V1`–`V28`, `Amount` e `Class`. `Class` vale 1 na fraude e 0 na transação legítima. `V1`–`V28` são componentes de PCA; `Time` e `Amount` não passaram por esse PCA. Nesta leitura há 0 nulos, `Amount` vai de 0 a 25.691,16 (1.825 valores zero) e `Time` vai de 0 a 172.792 segundos.

A assimetria de `Amount` é 16,9777. Depois de `log1p`, cai para 0,1627. Na tabela já sem cópia exata, a mediana do valor é 22,00 nas legítimas e 9,82 nas fraudes; a média é 88,41 e 123,87. O gráfico de hora do dia é só descrição da janela de dois dias. O arquivo não tem identificador de cliente nem de cartão, então não há sequência comportamental por pessoa.

## De onde vêm os dados, e o que foi conferido

O repositório, no início deste trabalho, tinha só o título. Não havia notebook da aula, link da Aula 1 nem código de instrutor. O link da aula do curso na DIO permanece desconhecido e não está confirmado.

A fonte pública equivalente usada aqui é o tutorial oficial do TensorFlow, [Classification on imbalanced data](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data), e o CSV que esse tutorial hospeda:

`https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`

O notebook baixa essa URL, sem conta e sem credencial. O código do tutorial é Apache 2.0. A página descreve o arquivo como o dataset do Kaggle coletado pela Worldline e pelo MLG-ULB. Ela não declara uma licença separada para o CSV.

O dataset de origem está em [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), organização Machine Learning Group - ULB. O JSON-LD público da página nomeia a licença como "Database: Open Database, Contents: Database Contents" e aponta [http://opendatacommons.org/licenses/dbcl/1.0/](http://opendatacommons.org/licenses/dbcl/1.0/) (Database Contents License, DbCL 1.0). A mesma página pede a citação abaixo. O rótulo curto da Kaggle junta "Open Database" e "Database Contents". O URL estruturado publicado na página leva à DbCL 1.0, não ao URL da Open Database License (`http://opendatacommons.org/licenses/odbl/1.0/`). Este texto registra o nome e o URL que a página publica.

Citação pedida na página do dataset:

Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. In *2015 IEEE Symposium Series on Computational Intelligence* (CIDM).

Conferência desta execução, em 7 de novembro de 2026:

| Fingerprint | Valor |
| --- | --- |
| sha256 do arquivo baixado | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |
| bytes | 150.828.752 |
| linhas / colunas | 284.807 / 31 |
| legítimas / fraudes | 284.315 / 492 |
| sha256 canônico da tabela carregada | `1700322b377ac8340ab6b75f22b974944fbcaf5b4f0daf3d5e8f620d96313026` |
| sha256 canônico depois da limpeza | `904d4d6a670f607d95ac7698ec70efc9bdd44cd527ecacfde8e09a46a0fe3491` |
| linhas depois da limpeza | 283.726 |

O sha256 canônico da tabela carregada já difere do sha256 do arquivo porque `to_csv` não reproduz os bytes originais. A limpeza muda de novo o hash e a contagem: 284.807 linhas passam a 283.726.

Não há dado sensível de cartão, nome, documento ou pagamento. O material é o dataset público anonimizado.

## O que este estudo faz em relação ao enunciado

O núcleo pedido está no notebook: EDA, `log1p` em `Amount`, `StandardScaler`, split estratificado sem vazamento de transação exata, baseline que prevê sempre a classe legítima, regressão logística, floresta aleatória e XGBoost com pesos, limiar por F2 na validação, métricas da classe 1, curvas ROC e precision–recall, SHAP real da árvore escolhida e as limitações abaixo.

Diferenças explícitas, sem inventar um código de instrutor que não estava no repositório:

- Reamostragem ficou de fora. O desbalanceamento entra por `class_weight` e por `scale_pos_weight`, calculados só no treino.
- Os hiperparâmetros são fixos. Não houve busca.
- O tutorial do TensorFlow descarta `Time` e treina uma rede Keras. Aqui `Time` permanece, padronizado, e os modelos são logística, floresta e XGBoost.
- As 1.825 transações com valor zero permanecem. Retirá-las mudaria a população publicada.
- Há testes automatizados das regras de split, escala e limiar em `tests/test_invariantes.py`.

## Como reproduzir

Ambiente desta execução: Python 3.12.3, Linux x86_64. Pacotes diretos:

| Pacote | Versão |
| --- | --- |
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| xgboost | 3.4.1 |
| shap | 0.54.1 |
| scipy | 1.18.1 |
| nbclient | 0.11.0 |
| nbconvert | 7.17.2 |
| ipykernel | 7.4.0 |
| pytest | 9.2.0 |

`requirements-lock.txt` é o `pip freeze` completo dessa execução, incluindo as dependências transitivas. O wheel do XGBoost 3.4.1 puxou `nvidia-nccl-cu13` neste Linux; o treino usou `device="cpu"`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name fraude-cartao --display-name "Python (fraude-cartao)"
.venv/bin/python -m pytest -q
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=fraude-cartao \
  notebooks/deteccao_fraude_cartao.ipynb
```

O CSV vai para `data/creditcard.csv`, que o `.gitignore` exclui. Se o arquivo local já tiver o sha256 acima, o notebook não baixa de novo. A semente é 42.

Nesta máquina o treino do conjunto inteiro levou 0,5 s na logística (26 iterações), 17,8 s na floresta (170.235 linhas) e 0,8 s no XGBoost.

## Preparação

Havia 1.081 cópias exatas de linha inteira (1.854 linhas em 773 grupos). Desses extras, 19 eram fraude e 1.062 eram legítimas. Nenhuma duplicata de atributos tinha rótulos conflitantes, e depois da remoção não sobrou duplicata de atributos. A decisão foi `drop_duplicates(keep="first")` antes do sorteio. Nenhum grupo precisou ser forçado para o treino. A sobreposição de vetores de atributos entre treino, validação e teste foi 0, 0 e 0.

O split estratificado, semente 42, sobre as 283.726 linhas limpas:

| Partição | Linhas | Fraudes | Legítimas |
| --- | ---: | ---: | ---: |
| Treino (60%) | 170.235 | 284 | 169.951 |
| Validação (20%) | 56.745 | 94 | 56.651 |
| Teste (20%) | 56.746 | 95 | 56.651 |

`log1p(Amount)` não estima parâmetro. O `StandardScaler` viu 170.235 linhas, o tamanho do treino. A média do treino padronizado fica em 0 e o desvio em 1. A maior média absoluta na validação, sem reajuste, foi 0,0123; no teste, 0,0116. O modelo recebe `Time_z`, `V1_z`–`V28_z` e `Amount_log1p_z`.

`scale_pos_weight` do XGBoost é 169.951 / 284 = 598,419014, só com a contagem do treino.

## Protocolo de decisão

Para cada modelo, o limiar da validação maximiza o F2 (beta = 2):

F2 = 5 · precisão · recall / (4 · precisão + recall).

Recall pesa mais, porque deixar passar uma fraude custa mais do que abrir um alerta a mais. A precisão continua na fórmula. No empate, fica o maior limiar. O modelo escolhido é o de maior F2 de validação nesse limiar; o desempate seria average precision e, depois, ROC AUC. O teste não entra na escolha.

O escore com peso de classe ordena alertas. Ele não é uma probabilidade calibrada, e o corte 0,5 não significa "50% de chance".

Average precision é a média das precisões ponderada pelo ganho de recall, sem interpolar. A área trapezoidal liga os pontos da curva por retas. No dummy, as duas divergem de forma visível: average precision igual à prevalência (0,001657 na validação) e área trapezoidal 0,500828. No XGBoost do teste a diferença é pequena (0,823784 − 0,823610 = 0,000174), e mesmo assim os dois números são reportados separados. A prevalência da partição é a linha de base da curva precision–recall. A diagonal é a linha de base da ROC.

## Métricas da execução

Validação, usada para escolher. O corte "F2" é o limiar travado; o corte "0,5" é a referência.

| Modelo | Corte | Limiar | Precisão | Recall | F1 | F2 | AP | Trapézio | ROC AUC | TP | FP | FN | Alertas / mil |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy sempre legítima | 0,5 | 0,500000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0017 | 0,5008 | 0,5000 | 0 | 0 | 94 | 0,00 |
| Dummy sempre legítima | F2 | 0,000000 | 0,0017 | 1,0000 | 0,0033 | 0,0082 | 0,0017 | 0,5008 | 0,5000 | 94 | 56.651 | 0 | 1.000,00 |
| Regressão logística | 0,5 | 0,500000 | 0,0529 | 0,9043 | 0,1000 | 0,2144 | 0,7894 | 0,8084 | 0,9801 | 85 | 1.521 | 9 | 28,30 |
| Regressão logística | F2 | 0,999784 | 0,7941 | 0,8617 | 0,8265 | 0,8473 | 0,7894 | 0,8084 | 0,9801 | 81 | 21 | 13 | 1,80 |
| Floresta aleatória | 0,5 | 0,500000 | 0,8421 | 0,8511 | 0,8466 | 0,8493 | 0,8508 | 0,8505 | 0,9726 | 80 | 15 | 14 | 1,67 |
| Floresta aleatória | F2 | 0,478621 | 0,8438 | 0,8617 | 0,8526 | 0,8581 | 0,8508 | 0,8505 | 0,9726 | 81 | 15 | 13 | 1,69 |
| XGBoost | 0,5 | 0,500000 | 0,8632 | 0,8723 | 0,8677 | 0,8705 | 0,8856 | 0,8855 | 0,9735 | 82 | 13 | 12 | 1,67 |
| XGBoost | F2 | 0,567120 | 0,8723 | 0,8723 | 0,8723 | 0,8723 | 0,8856 | 0,8855 | 0,9735 | 82 | 12 | 12 | 1,66 |

O dummy no corte que maximiza o F2 marca todas as linhas: recall 1, precisão igual à prevalência, F2 = 0,0082 e 1.000 alertas por mil. Esse é o extremo de perseguir só o recall. A logística com `class_weight="balanced"`, no corte 0,5, abre 1.606 alertas na validação (precisão 0,0529, F2 0,2144). O limiar que maximiza o F2 dela sobe para 0,999784 e o F2 vai a 0,8473. O escore dela se concentra perto de 1; o corte 0,5 não é um ponto operacional útil.

Modelo escolhido na validação, antes do teste: **XGBoost**, limiar **0,5671201348304749**. F2 de validação 0,872340, acima da floresta (0,858051) e da logística (0,847280).

Teste, lido uma vez nesse par. Os outros modelos não foram reordenados pelo teste.

| Leitura | Precisão | Recall | F1 | F2 | AP | Trapézio | ROC AUC | TP | FP | FN | TN | Alertas | Alertas / mil | Acurácia |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost no limiar travado | 0,8523 | 0,7895 | 0,8197 | 0,8013 | 0,8238 | 0,8236 | 0,9686 | 75 | 13 | 20 | 56.638 | 88 | 1,55 | 0,9994 |
| XGBoost no corte 0,5 | 0,8172 | 0,8000 | 0,8085 | 0,8034 | 0,8238 | 0,8236 | 0,9686 | 76 | 17 | 19 | 56.634 | 93 | 1,64 | 0,9994 |
| Dummy no corte 0,5 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0017 | 0,5008 | 0,5000 | 0 | 0 | 95 | 56.651 | 0 | 0,00 | 0,9983 |

Relatório de classificação do teste no limiar travado: fraude com precisão 0,8523, recall 0,7895 e F1 0,8197 (suporte 95); legítima com precisão 0,9996 e recall 0,9998.

No teste, o corte 0,5 teve F2 0,803383 e o limiar travado teve F2 0,801282. A diferença cabe na variação de 95 fraudes. A escolha permanece o limiar da validação. Trocar depois de ver o teste seria escolher pelo teste.

Carga do limiar oficial: 88 alertas em 56.746 transações (1,55 por mil), 13 falsos positivos (0,23 por mil) e 20 fraudes não alertadas. De cada alerta, cerca de 85% são fraude nesta partição.

Hiperparâmetros fixos, treino completo: logística `class_weight="balanced"`, `lbfgs`, 1.000 iterações máximas; floresta com 160 árvores, profundidade 12, folha mínima 8, `max_features="sqrt"`, `class_weight="balanced_subsample"`; XGBoost com 200 árvores, profundidade 4, `learning_rate=0,1`, `min_child_weight=4`, `subsample=0,8`, `colsample_bytree=0,8`, `tree_method="hist"`.

## O que o SHAP mostrou

O modelo explicado é o XGBoost escolhido na validação. O TreeSHAP foi calculado de fato. A amostra global tem 1.594 linhas da validação: as 94 fraudes e 1.500 legítimas, semente 42. A fraude está sobrerrepresentada para aparecer no gráfico. Essa amostra não retreina o modelo e não mexe no limiar.

O espaço foi conferido contra `predict_proba`: a sigmoide de (valor-base + soma dos SHAP) reproduz o escore, com erro máximo 4,03×10⁻⁷ na amostra e 5,97×10⁻⁸ nos casos locais. O espaço é log-odds, antes da sigmoide. O valor-base na amostra é 2,469699. A sigmoide desse valor é cerca de 0,922. Esse piso alto acompanha o `scale_pos_weight` de 598,42, que trata a fraude como muito mais comum na perda do que ela é nos dados. O valor-base não é a prevalência (0,17%) e o escore não é uma probabilidade calibrada de fraude.

Na amostra misturada, a média de |SHAP| é liderada por `V14_z` (2,269), `V4_z` (1,791), `V12_z` (1,009), `V3_z` (0,762) e `V11_z` (0,697). Nas 94 fraudes da amostra a ordem começa por `V14_z`, `V12_z`, `V10_z`, `V4_z` e `V3_z`. Nas 1.500 legítimas, por `V14_z`, `V4_z`, `V12_z`, `V3_z` e `V11_z`. O ganho interno do XGBoost também coloca `V14_z` na frente (0,423 do ganho normalizado), seguido de `V4_z` e `V10_z`. São componentes de PCA padronizados. O gráfico não recupera idade, estabelecimento ou qualquer traço pessoal. A associação é com o log-odds deste modelo.

Os três casos locais saem do teste, depois da métrica. A regra é o escore mais próximo da mediana do grupo, com desempate no menor índice.

- Verdadeiro positivo, índice 2.146 no teste, fraude, escore 0,9998, `Amount` original 0,00. Esse zero é o valor desta transação, escolhida pela mediana do escore, e não um perfil geral das fraudes. `V14_z` = −11,005 contribui +3,580 em log-odds. A saída SHAP é 8,677, partindo da base 2,470. `Amount_log1p_z` = −1,905 aparece na cascata com SHAP +0,353: o modelo vê o valor depois de `log1p` e da padronização do treino, não o euro bruto. `log1p(0) = 0`, e a padronização desloca esse zero para −1,905.
- Falso positivo, índice 4.051, legítima, escore 0,6587, `Amount` original 10,61. `V14_z` = −7,068 empurra o log-odds em +2,952; outras componentes puxam de volta. A saída fica em 0,657, ainda acima do limiar 0,567.
- Falso negativo, índice 22.495, fraude, escore 0,0012, `Amount` original 105,99. As maiores contribuições são negativas (`V11_z` −1,790, `V17_z` −0,769, `V8_z` −0,738). A saída vai a −6,762, bem abaixo do limiar.

Figuras: `04_limiar_f2_validacao.png`, `05_f2_validacao.png`, `06_roc_validacao.png`, `07_pr_validacao.png`, `08_matriz_confusao_teste.png`, `09_roc_pr_teste.png`, `10_shap_barra.png`, `11_shap_beeswarm.png`, `12_shap_por_classe.png`, `13_importancia_arvore.png`, `14_coeficientes_logistica.png`, `15_shap_waterfall_tp.png`, `15_shap_waterfall_fp.png`, `15_shap_waterfall_fn.png`.

## Limitações

A divisão embaralha as transações e preserva a proporção de fraudes. O número descreve este recorte histórico de dois dias, tratado como IID. Ele não mede produção futura, drift nem uma fila real de investigação. Um corte temporal seria outra pergunta e não foi usado como segundo teste. Validação escolheu modelo e limiar no mesmo bloco, então o F2 de validação é um pouco otimista. Com 95 fraudes no teste, precisão e recall ainda se movem se a semente mudar.

`V1`–`V28` são PCA publicados assim por confidencialidade. Importância e SHAP não viram nome, país, renda ou traço pessoal, e não provam fraude nem causa. O peso de classe muda a perda; o escore ordena a fila de alerta.

## Testes

`tests/test_invariantes.py` cobre, com dados sintéticos, a fórmula do F2, o limiar que não marca tudo como fraude, o corte 0,5 quando ele perde para outro ponto, a concordância da varredura com a regra `escore >= limiar`, a divergência possível entre average precision e área trapezoidal, a remoção da duplicata antes do split, o grupo com rótulo conflitante inteiro no treino, a reprodutibilidade do split, o scaler ajustado só no treino sobre `log1p(Amount)`, o `scale_pos_weight` medido só no treino, e a regra dos casos locais. Se `data/creditcard.csv` estiver presente, um teste confere o sha256 público.

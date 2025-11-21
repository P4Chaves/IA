# **Requisitos do Projeto — OSMI Mental Health in Tech (2018)**

**Integrantes (RM):**  
- **Pedro Chaves — RM 553988**  
- **Iago Diniz — RM 553776**  
- **Lucas Garcia — RM 554070**

---

## **Resumo rápido (o que vamos fazer)**
**Problema:** prever, a partir das respostas do questionário OSMI 2018, **a probabilidade de a pessoa ter buscado tratamento de saúde mental** (**Yes/No**).  
**Abordagem:** pipeline de **ML clássico com Regressão Logística** (com **L1/L2**), **padronização + one-hot**, **validação cruzada (k=5) com GridSearchCV**, avaliação em **hold-out** e **interpretação pelas features/coeficientes**.

---

## **1) Escolha do conjunto de dados**
- **Base:** *OSMI – Mental Health in Tech Survey (2018)*, arquivo **`osmi.csv`**.
- **Domínio:** **saúde mental no trabalho** em tecnologia (benefícios, abertura da empresa, regime de trabalho, estigma, etc.).
- **Alvo (binário):** coluna relacionada a **busca/necessidade de tratamento** — detectada por nome contendo **“treat”** (ex.: `treatment`, `seek_treatment`, `mh_treatment`).
  - **Mapeamento:** **Yes/True/Y/1 → 1** e **No/False/N/0 → 0**.
- **Features:** variáveis **numéricas** e **categóricas** do questionário.
- **Exclusões:** **identificadores** e **texto livre** (ex.: `Timestamp`, `ID`, `comments`) e categóricas com **altíssima cardinalidade** (evitar explosão de dummies).

---

## **2) Enunciado do problema**
> **Dado o perfil do(a) profissional e o contexto organizacional reportado no OSMI 2018, queremos** **prever a probabilidade** de a pessoa **ter buscado tratamento de saúde mental** (**Yes/No**), construindo um **classificador** com **bom desempenho e interpretabilidade**, capaz de **evidenciar fatores associados** ao comportamento de busca por tratamento, respeitando **ética** e **privacidade**.

---

## **3) Estratégia (pipeline de ML)**
1. **Pré-processamento**
   - **Imputação:** **numéricas → mediana**; **categóricas → mais frequente**.
   - **Padronização** das numéricas com **`StandardScaler`** (estabilidade e penalização justa).
   - **One-Hot Encoding** para categóricas (`handle_unknown="ignore"`).
   - **Remoção** de texto livre/IDs e categóricas com **> 50 categorias**.
2. **Modelo base:** **`LogisticRegression`**
   - Testaremos **regularização L1 (Lasso)** e **L2 (Ridge)**.
   - **Solver:** `liblinear` (suporta L1/L2 em binário).
3. **Validação & seleção de hiperparâmetros**
   - **`GridSearchCV`** com **k-fold estratificado (k=5)**.
   - **Grade:** `penalty ∈ {L1, L2}` e `C ∈ {0.01, 0.1, 1, 10, 100}`.
4. **Avaliação final (hold-out)**
   - **Split:** **treino/teste = 80/20**, **estratificado**.
   - **Métricas no teste:** **Acurácia, Precisão, Recall, F1-score** e **Matriz de confusão**.  
     *(Opcional: **ROC-AUC** e **curva de calibração**).*
5. **Interpretação**
   - **Comparar coeficientes** de **L1 vs L2** e **listar features zeradas** pelo L1 (seleção automática).
   - Discutir como **políticas/ambiente** aparecem nas **features mais relevantes**.

---

## **4) Justificativa do uso da ferramenta/modelo**
- **Por que Regressão Logística?**  
  - **Alvo binário** (Yes/No) e necessidade de **probabilidades** → a logística é **natural** e bem fundamentada (log-odds).
  - **Interpretabilidade:** coeficientes indicam **direção e magnitude** do efeito (após padronização/one-hot).
  - **Regularização embutida:** **L1** (seleção de variáveis/explicabilidade) e **L2** (estabilidade com **multicolinearidade**).
  - **Simplicidade + desempenho**: baseline **forte e transparente** para dados tabulares com muitas dummies.
- **Por que `StandardScaler` + `OneHotEncoder` + `ColumnTransformer`?**  
  - **Evita vazamento** (fit só no **treino**, reutilizado no pipeline).
  - **Escalas comparáveis** → regularização **justa** e otimização **estável**.
  - **One-hot robusto** para categorias **novas** no teste (`handle_unknown="ignore"`).
- **Por que `GridSearchCV (k=5)`?**  
  - **Seleção objetiva** de hiperparâmetros (penalty/C) com **estratificação** e **reprodutibilidade**.
  - **Reduz risco** de overfitting de configuração e **mede variabilidade** entre folds.

---

## **5) Validação cruzada e regularização (aplicação adequada)**
- **Divisão correta:** **reservar 20%** para **teste**; rodar o **GridSearchCV apenas no treino** com **`StratifiedKFold(k=5)`**.
- **Regularização:**
  - **L2 (Ridge):** **encolhe** coeficientes, lida bem com **multicolinearidade** e **raramente zera**.
  - **L1 (Lasso):** induz **esparsidade** (**coeficientes = 0**), fazendo **seleção de variáveis** e aumentando a **interpretabilidade**.
- **Hiperparâmetro `C`:** controla a **força da penalização** (**menor C = penalização mais forte**).
- **Boas práticas:**
  - Se houver **desbalanceamento**, avaliar `class_weight="balanced"` **dentro da busca**.
  - Checar **calibração** (curva de confiabilidade) se as **probabilidades** forem usadas em decisão.

---

## **6) Requirements (ambiente e bibliotecas)**
- **Versão de Python sugerida:** **3.10+**  
- **Pacotes principais:**
  numpy
  pandas
  scikit-learn
  matplotlib
- **Instalação rápida:**
pip install -r requirements.txt
# ou
pip install numpy pandas scikit-learn matplotlib

---

# **Execução, Critérios de Sucesso e Diretrizes Éticas**

## **7) Como rodar (reprodutibilidade)**
1. Coloque o arquivo **`osmi.csv`** na raiz do projeto ou ajuste o caminho no notebook.  
2. Execute o notebook na ordem: **A1 (Carregar/Preparar) → A2 (EDA) → A3 (GridSearch CV L1/L2) → A4 (Teste & Métricas) → A5 (L1×L2 & Interpretação)**.  
3. Use **seeds fixos** (`random_state=42`) para splits e validação cruzada.  
4. Registre no README/PDF: **melhores hiperparâmetros**, **métricas do teste** e **features zeradas** (L1).

## **8) Critérios de sucesso**
- **Validação (média 5-fold):** **acurácia ≥ X%** (preencher após execução) com **baixa variância** entre folds.  
- **Teste (hold-out):** **acurácia** e **F1** próximas/consistentes com a validação.  
- **Interpretação:** **top coeficientes** e **features zeradas (L1)** coerentes com o domínio (políticas/ambiente).  
- **Reprodutibilidade:** pipeline único (**`ColumnTransformer` + `LogisticRegression`**) e versões de pacotes registradas.

## **9) Observações éticas e de privacidade**
- **Minimização de dados:** usar apenas colunas necessárias; excluir **texto livre** e **identificadores**.  
- **Viés/justiça:** verificar se o desempenho varia por **grupos** (se houver atributos sensíveis).  
- **Uso responsável:** as **probabilidades** não são **diagnóstico**; suportam decisões de políticas de **apoio e bem-estar**.

## **Conclusões**

### **10) Conclusão**
Mostramos que **é viável prever, com alta qualidade, quem buscou tratamento de saúde mental** no OSMI-2018 usando um pipeline simples e regularizado. O **GridSearchCV (k=5)** escolheu **Regressão Logística com L2 (C=0,1)** (**acurácia CV = 0,8929**). No **teste**, obtivemos **Acc 0,9643 | Prec 0,9615 | Recall 1,0000 | F1 0,9804 | AUC 0,9867**, com matriz `[[2, 1], [0, 25]]` (sem falsos negativos). A L2 foi mais estável que a L1 em validação, embora a **L1 tenha zerado 430 variáveis**, indicando que o sinal está concentrado em um subconjunto de features. Os **fatores organizacionais e culturais** (importância dada à saúde mental, reação de colegas, impacto na carreira, benefícios e facilidade/dificuldade de licença) aparecem como **preditores mais influentes**. Limitação: **poucos negativos no teste**; apesar disso, o **AUC alto** e a **validação cruzada** sustentam a generalização. Em termos práticos, **fortalecer benefícios, normalizar a conversa sobre saúde mental e facilitar licenças** são frentes plausíveis — sempre com uso **ético** do modelo (apoio a políticas, não decisões individuais sensíveis).

### **11) Conclusão técnica (resumo numérico)**
- **Melhor configuração (CV):** `penalty = L2`, `C = 0.1`  → **acurácia média CV = 0,8929**  
- **Teste (20%):** **Acc 0,9643 | Prec 0,9615 | Rec 1,0000 | F1 0,9804 | AUC 0,9867**  
- **Matriz de confusão (TESTE):** `[[TN=2, FP=1], [FN=0, TP=25]]`  
- **Sparsidade (L1):** **430** coeficientes zerados (boa seleção, mas L2 generalizou melhor)  
- **Sinais mais fortes (pós-processamento):** importância dada pela empresa à saúde mental; reação esperada dos colegas; impacto percebido na carreira; **benefícios de saúde mental**; **dificuldade para pedir licença médica**; suporte da indústria  
- **Observação/risco:** **amostra de teste com poucos negativos** → interpretar com cautela; **CV + AUC** ajudam a validar a robustez  
- **Recomendação prática:** focar em **benefícios**, **clima/estigma** e **processos de licença**; monitorar **calibração** e **viés** entre subgrupos


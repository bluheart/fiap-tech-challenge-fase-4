# fiap-tech-challenge-fase-4

## Governança e LGPD

### 1. Natureza dos Dados e Tratamento de PII

O dataset utilizado (german_credit_data.csv) é derivado do South German Credit Dataset (UCI Machine Learning Repository, 2019), que consiste em uma amostra estratificada de 1.000 créditos concedidos entre 1973 e 1975 por um banco regional alemão . Os dados são historicamente anonimizados e disponibilizados publicamente para fins de pesquisa.

Não há PII (Informação Pessoal Identificável) real neste projeto. As variáveis disponíveis são atributos socioeconômicos e financeiros (idade, valor do crédito, duração, finalidade, categoria de conta poupança, sexo, estado civil, entre outros), sem nome, CPF, endereço, telefone ou qualquer identificador direto de titulares.

Medidas de Privacy by Design adotadas (alinhadas ao art. 46, §2º da LGPD ):
Princípio	Implementação no projeto
Minimização	Apenas as features necessárias ao modelo de risco são utilizadas. Colunas como telephone ou foreign_worker foram descartadas por não agregarem poder preditivo relevante e aumentarem a superfície de exposição.
Finalidade	Os dados são usados exclusivamente para treino do classificador de risco de crédito e demonstração de detecção de drift. Nenhum uso secundário é realizado.
Segurança desde a concepção	O pipeline não persiste PII em logs. O logs/pipeline.jsonl registra apenas metadados de execução (step, status, rows, duration). O mlflow.db armazena parâmetros e métricas do modelo, não dados brutos.
Transparência	Este README documenta todas as etapas do tratamento. O código-fonte é público no repositório GitHub.


### 2. Base Legal para Decisão de Crédito

A base legal primária para o tratamento de dados pessoais em contexto de proteção ao crédito é o art. 7º, X da LGPD ("proteção do crédito"), que sustenta a formação de histórico de crédito, manutenção de cadastros e uso de score . O Superior Tribunal de Justiça (STJ) consolidou entendimento de que o credit scoring é prática lícita e dispensa consentimento prévio do consumidor, desde que respeitados os limites de privacidade e transparência .

No contexto deste projeto acadêmico, a base legal aplicável é simulada, uma vez que os dados são anonimizados e públicos. Para fins de demonstração, assumimos:

    Base legal principal: art. 7º, X (proteção ao crédito) — para o treino do modelo de risco.

    Base legal complementar: art. 7º, IX (legítimo interesse) — para o monitoramento contínuo de drift, desde que realizado teste de balanceamento e mantidas salvaguardas técnicas (que estão implementadas: logs sem PII, métricas agregadas, sem reidentificação).

    Decisões automatizadas: o modelo produz uma probabilidade de inadimplência (prediction_proba), mas não toma decisões autonomamente. A saída é um insumo para análise humana. Caso o modelo fosse implantado em produção real, o art. 20 da LGPD exigiria direito à revisão de decisões automatizadas, o que este projeto documenta como requisito para implantação futura .

### 3. Plano de Retenção

A LGPD não fixa prazos específicos; a retenção deve observar a finalidade e as hipóteses do art. 16 (cumprimento de obrigação legal, estudo por órgão de pesquisa, uso exclusivo do controlador com anonimização) . Para este projeto, o plano é:
Artefato	Prazo de retenção	Justificativa
data/reference/credit_reference.csv	Permanente (versionado no Git)	Dados públicos e anonimizados; necessários para reprodutibilidade do baseline.
data/production/credit_production.csv	Até 90 dias após a geração	Dataset sintético de drift; descartável. Pode ser regenerado por simulate_drift.py.
models/baseline_credit_model.joblib	Permanente (versionado)	Artefato do modelo; necessário para detecção de drift e auditoria.
logs/pipeline.jsonl	90 dias (rotação manual ou via logrotate)	Logs operacionais; não contêm PII. Após 90 dias, agregados em reports/metrics/pipeline_health.json.
reports/drift_report.html	12 meses	Evidência de monitoramento para auditoria. Substituído a cada nova execução.
mlflow.db (SQLite)	12 meses	Metadados de experimentos (parâmetros, métricas, run_id). Sem dados brutos.

Eliminação: ao final do projeto acadêmico, os datasets de produção e logs podem ser removidos. O modelo e os relatórios de drift são mantidos para fins de avaliação da disciplina.


### 4. Governança e Accountability

O projeto adota as seguintes práticas de governança, alinhadas ao que a ANPD recomenda para operações de crédito :

    Rastreabilidade: cada execução do pipeline gera um run_id no MLflow e uma linha correspondente em logs/pipeline.jsonl. O dashboard consolida ambos.

    Auditoria de modelo: o MLflow registra parâmetros, métricas (AUC, PR-AUC) e artefatos. O Model Registry permite versionamento e promoção controlada.

    Detecção de drift como controle contínuo: o detect_drift.py executa Evidently com testes estatísticos (KS, chi-square) e gera alertas em drift_metrics.json. Esse é o mecanismo de monitoramento pós-implantação exigido pelo princípio da prestação de contas.

    Documentação de decisões: este README consolida as escolhas técnicas, bases legais e limitações. O histórico de commits semânticos (feat:, fix:, docs:) permite rastrear a evolução do projeto.

### 5. Limitações e Riscos Residuais
Risco	Mitigação
Dataset histórico (1973–1975) pode conter vieses de época (ex.: tratamento desigual por sexo)	Documentado no README. O modelo é uma demonstração técnica, não um sistema de decisão real.
Drift simulado não captura todos os cenários reais	O pipeline permite adicionar novos cenários em simulate_drift.py. O Evidently cobre múltiplos testes estatísticos.
Ausência de revisão humana no fluxo	Documentado como requisito para implantação real (art. 20 LGPD).
SQLite pode bloquear em execuções concorrentes	Para o escopo do projeto (execuções sequenciais), não é um problema. Documentado como limitação conhecida.

Declaração de conformidade: este projeto utiliza dados públicos e anonimizados, não trata PII real, e implementa controles de privacidade desde a concepção (Privacy by Design) conforme art. 46, §2º da LGPD. O plano de retenção e as bases legais estão documentados para fins de demonstração acadêmica.

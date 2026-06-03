# Databricks notebook source

# MAGIC %md
# MAGIC # AutoPremium Japan カスタマーサービスエージェント
# MAGIC ## ノートブック 03: デプロイ・MLflow評価・LLM as Judge
# MAGIC
# MAGIC このノートブックでは以下を実施します:
# MAGIC 1. エージェントをMLflowに登録 → Unity Catalogに登録
# MAGIC 2. `databricks.agents.deploy()` でサービングエンドポイントとレビューアプリをデプロイ
# MAGIC 3. **MLflow Evaluate + LLM as Judge** で自動品質評価を実施
# MAGIC    - 回答の正確性（Answer Correctness）
# MAGIC    - 根拠との整合性（Faithfulness / 幻覚検出）
# MAGIC    - 関連性（Relevance）
# MAGIC    - ティア適合性・安全性認識（カスタムジャッジ）
# MAGIC 4. 評価結果の可視化・レビューアプリの案内

# COMMAND ----------

# MAGIC %pip install databricks-agents mlflow databricks-sdk databricks-vectorsearch openai "typing_extensions>=4.12.0" --upgrade -q

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 設定読み込み・初期化
# MAGIC
# MAGIC 設定値は `config.py` で一元管理しています。カタログ名等の変更は `config.py` のみ編集してください。

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import json
import os

import mlflow
import pandas as pd
from databricks import agents
from databricks.sdk import WorkspaceClient

DATABRICKS_HOST  = spark.conf.get("spark.databricks.workspaceUrl")
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

mlflow.set_experiment(EXPERIMENT_NAME)
w = WorkspaceClient()

print(f"Workspace      : https://{DATABRICKS_HOST}")
print(f"Catalog        : {CATALOG}")
print(f"Agent Model    : {AGENT_MODEL_NAME}")
print(f"Endpoint       : {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. エージェントをMLflowに登録

# COMMAND ----------

context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
notebook_dir = os.path.dirname(context.notebookPath().get())
agent_file_path = f"/Workspace{notebook_dir}/agent.py"

input_example = {
    "messages": [{"role": "user", "content": "顧客ID C001です。私の車のオイル交換について教えてください。"}]
}

with mlflow.start_run(run_name="auto_service_agent_v1") as run:
    logged_model = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=agent_file_path,
        pip_requirements=["mlflow", "databricks-sdk", "databricks-vectorsearch", "openai"],
        input_example=input_example,
        resources=[
            mlflow.models.resources.DatabricksVectorSearchIndex(index_name=FAQ_INDEX_NAME),
            mlflow.models.resources.DatabricksVectorSearchIndex(index_name=CASE_INDEX_NAME),
            mlflow.models.resources.DatabricksServingEndpoint(endpoint_name=LLM_MODEL),
        ]
    )
    model_uri = logged_model.model_uri

print(f"✅ MLflow にログしました: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Unity Catalog にモデルを登録

# COMMAND ----------

registered = mlflow.register_model(model_uri=model_uri, name=AGENT_MODEL_NAME)
model_version = registered.version

mlflow.tracking.MlflowClient().set_registered_model_alias(
    name=AGENT_MODEL_NAME, alias="champion", version=model_version
)
print(f"✅ UC 登録完了: {AGENT_MODEL_NAME} v{model_version} (alias: champion)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. エンドポイントとレビューアプリをデプロイ

# COMMAND ----------

deployment = agents.deploy(
    model_name=AGENT_MODEL_NAME,
    model_version=model_version,
    scale_to_zero=True,
    endpoint_name=ENDPOINT_NAME,
)

print(f"✅ デプロイ完了!")
print(f"")
print(f"📡 Serving Endpoint : https://{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}")
print(f"")
print(f"🔍 Review App URL   : {deployment.review_app_url}")
print(f"")
print(f"レビューアプリでは以下が可能です:")
print(f"  - エージェントとの会話テスト")
print(f"  - 各回答への 👍/👎 フィードバック収集")
print(f"  - フィードバックデータの蓄積（評価データセット改善に活用）")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 評価データセットの準備

# COMMAND ----------

eval_data = json.load(open(f"{VOLUME_PATH}/eval_dataset.json", encoding="utf-8"))

eval_df = pd.DataFrame([{
    "request":          json.dumps(item["request"], ensure_ascii=False),
    "expected_response": item.get("expected_response", ""),
    "ground_truth_context": item.get("ground_truth_context", ""),
    "question": item["request"]["messages"][0]["content"]
} for item in eval_data])

print(f"✅ 評価データセット: {len(eval_df)} 件")
display(eval_df[["question", "expected_response"]].head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. LLM as Judge カスタム指標の定義

# COMMAND ----------

from mlflow.metrics.genai import make_genai_metric, EvaluationExample

customer_tier_compliance = make_genai_metric(
    name="customer_tier_compliance",
    definition=(
        "顧客の会員ティア（スタンダード/シルバー/ゴールド）に適した特典・サービス情報が "
        "正確に提供されているかを評価する。誤ったティア情報を提供した場合は低スコア。"
    ),
    grading_prompt=(
        "以下の回答が、顧客の会員ティアに正確に対応した特典・サービス内容を案内しているか評価してください。\n\n"
        "5点: ティアに応じた特典が正確かつ完全に案内されている\n"
        "4点: ほぼ正確だが一部の特典情報が不足\n"
        "3点: ティア情報は示されているが正確性に疑問がある\n"
        "2点: ティア情報が不正確または混乱を招く\n"
        "1点: ティア情報に重大な誤りがある\n\n"
        "question: {question}\nground_truth: {ground_truth}\nresponse: {prediction}\n\n"
        "スコアと理由を日本語で回答してください。"
    ),
    examples=[
        EvaluationExample(
            input="顧客ID C001です。車検の割引はありますか？",
            output="田中太郎様はゴールド会員ですので、整備・車検費用に20%割引が適用されます。",
            score=5, justification="ゴールド会員の正確な割引率（20%）を案内している"
        ),
        EvaluationExample(
            input="顧客ID C002です。車検の割引はありますか？",
            output="シルバー会員ですので、整備費用20%割引が適用されます。",
            score=2, justification="シルバー会員の割引率は15%であり、20%は誤り"
        )
    ],
    model=f"endpoints:/{LLM_MODEL}",
    parameters={"temperature": 0.0},
    greater_is_better=True,
    max_workers=4
)

safety_urgency_detection = make_genai_metric(
    name="safety_urgency_detection",
    definition=(
        "ブレーキ不良・タイヤバースト等の安全に関わる問題を適切に緊急扱いし、"
        "即座の対応を促しているかを評価する。安全問題でない場合は5点とする。"
    ),
    grading_prompt=(
        "以下の回答が、安全に関わる問題を適切に緊急扱いしているか評価してください。\n"
        "安全問題でない場合は5点（評価対象外）としてください。\n\n"
        "5点: 緊急性を明確に伝え、即座のサービス対応を強く推奨している\n"
        "4点: 緊急性を示しているが具体的な行動指示が不十分\n"
        "3点: 問題を認識しているが緊急性の表現が弱い\n"
        "2点: 問題を軽く扱い緊急性を伝えていない\n"
        "1点: 安全問題を無視または誤情報を提供\n\n"
        "question: {question}\nresponse: {prediction}\n\nスコアと理由を日本語で回答してください。"
    ),
    examples=[
        EvaluationExample(
            input="ブレーキを踏むとペダルが奥まで沈みます",
            output="これは非常に危険な状態です。すぐに走行を停止し、ディーラーにご連絡ください。",
            score=5, justification="安全問題として緊急対応を強く推奨"
        ),
        EvaluationExample(
            input="ブレーキを踏むとペダルが奥まで沈みます",
            output="ブレーキの点検をお勧めします。次回の定期点検時にご相談ください。",
            score=1, justification="緊急問題を通常点検扱いしており非常に危険"
        )
    ],
    model=f"endpoints:/{LLM_MODEL}",
    parameters={"temperature": 0.0},
    greater_is_better=True,
    max_workers=4
)

print("✅ カスタム評価指標を定義しました: customer_tier_compliance / safety_urgency_detection")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. mlflow.evaluate() 実行（LLM as Judge）

# COMMAND ----------

import urllib.request

def predict_fn(request_json_str: str) -> str:
    """デプロイ済みエンドポイントに問い合わせして回答を返す"""
    url = f"https://{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    req = urllib.request.Request(
        url,
        data=json.dumps(json.loads(request_json_str)).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {DATABRICKS_TOKEN}"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    except Exception as e:
        return f"エラー: {str(e)}"

print("エージェントの回答を生成中...")
eval_df["response"] = eval_df["request"].apply(predict_fn)
print(f"✅ {len(eval_df)} 件の回答を生成しました")

# COMMAND ----------

from mlflow.metrics.genai import answer_correctness, answer_relevance, faithfulness

print("LLM as Judge 評価を実行中...（数分かかります）")

with mlflow.start_run(run_name="auto_service_agent_eval_v1") as eval_run:
    eval_results = mlflow.evaluate(
        data=eval_df,
        predictions="response",
        targets="expected_response",
        model_type="question-answering",
        extra_metrics=[
            answer_correctness(model=f"endpoints:/{LLM_MODEL}"),
            answer_relevance(model=f"endpoints:/{LLM_MODEL}"),
            faithfulness(model=f"endpoints:/{LLM_MODEL}"),
            customer_tier_compliance,
            safety_urgency_detection,
        ],
        evaluator_config={"col_mapping": {"inputs": "question", "context": "ground_truth_context"}}
    )

print(f"✅ 評価完了! Run ID: {eval_run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. 評価結果の分析

# COMMAND ----------

print("=" * 60)
print("📊 評価結果サマリー")
print("=" * 60)
for key, value in eval_results.metrics.items():
    if isinstance(value, float):
        bar = "█" * int(value) + "░" * (5 - int(min(value, 5)))
        print(f"  {key:<45}: {bar} {value:.2f}/5.00")

# COMMAND ----------

eval_table = eval_results.tables["eval_results_table"]
score_cols = [c for c in eval_table.columns if "/score" in c]
display_cols = ["question", "response"] + score_cols
display(eval_table[[c for c in display_cols if c in eval_table.columns]])

# COMMAND ----------

# スコアが低い回答（改善対象）
print("⚠️  スコア 3.5 未満の回答（改善候補）")
for col in score_cols:
    low = eval_table[eval_table[col] < 3.5][["question", col]]
    if len(low) > 0:
        print(f"\n指標: {col}")
        for _, row in low.iterrows():
            print(f"  質問: {row['question'][:80]}... | スコア: {row[col]:.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. MLflow ダッシュボード・リンク一覧

# COMMAND ----------

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
exp_url = f"https://{DATABRICKS_HOST}/#mlflow/experiments/{experiment.experiment_id}" if experiment else "（実験未作成）"

print("=" * 60)
print("🔗 リンク一覧")
print("=" * 60)
print(f"📡 Serving Endpoint : https://{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}")
print(f"🔍 Review App       : {deployment.review_app_url}")
print(f"📈 MLflow 実験       : {exp_url}")
print()
print("=" * 60)
print("🎯 デモシナリオ早見表")
print("=" * 60)
scenarios = [
    ("ゴールド会員の特典案内", "C001（田中太郎, ゴールド）", "私の車のオイル交換と車検の割引を教えてください"),
    ("故障診断（事例検索）",   "C002（鈴木花子, シルバー）", "エンジンからカラカラ音がします"),
    ("安全緊急対応",           "C007（中村浩介, ゴールド）", "ブレーキペダルが奥まで沈みます。走行できますか？"),
    ("EV充電トラブル",         "C003（山田一郎, スタンダード）", "急速充電ができないエラーが出ます"),
    ("リコール対応確認",       "C005（伊藤健二, シルバー）", "V006のリコールは費用がかかりますか？"),
]
for title, customer, question in scenarios:
    print(f"\n【{title}】")
    print(f"  顧客 : {customer}")
    print(f"  質問 : {question}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 完了 ✅
# MAGIC
# MAGIC | 機能 | 確認場所 |
# MAGIC |------|---------|
# MAGIC | エージェント動作 | Review App |
# MAGIC | トレース（ツール呼び出し詳細） | MLflow Experiments → Traces |
# MAGIC | LLM as Judge 評価スコア | MLflow Experiments → eval_results |
# MAGIC | フィードバックデータ | Unity Catalog → `{ENDPOINT_NAME}_payload` |

# Databricks notebook source

# MAGIC %md
# MAGIC # AutoPremium Japan カスタマーサービスエージェント
# MAGIC ## ノートブック 01: データ準備・ベクターサーチ設定
# MAGIC
# MAGIC このノートブックでは以下を設定します:
# MAGIC 1. Unity Catalog (カタログ・スキーマ・ボリューム) の作成
# MAGIC 2. FAQデータ・サービス事例をDelta Tableに格納（メダリオンアーキテクチャ）
# MAGIC 3. 顧客情報・車両情報をOnline Table（Feature Store）として設定
# MAGIC 4. Vector Search エンドポイント・インデックスの作成
# MAGIC
# MAGIC **想定されるユースケース**: 自動車ディーラー・メーカーのカスタマーサービスエージェント
# MAGIC - 顧客ティア（スタンダード/シルバー/ゴールド）に応じてパーソナライズされた回答を提供
# MAGIC - FAQ検索・サービス事例検索による正確な情報提供
# MAGIC - MLflowによる品質評価・モニタリング

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. パッケージインストール

# COMMAND ----------

# MAGIC %pip install databricks-sdk mlflow databricks-vectorsearch --upgrade -q

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
import shutil
import time

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
from databricks.sdk.service.vectorsearch import (
    EndpointType,
    VectorIndexType,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
)
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

w = WorkspaceClient()
print(f"Workspace : {w.config.host}")
print(f"Catalog   : {CATALOG}")
print(f"Schema    : {SCHEMA}")
print(f"LLM       : {LLM_MODEL}")
print(f"Embedding : {EMBEDDING_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Unity Catalog セットアップ

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"✅ カタログ・スキーマ・ボリューム作成完了: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. JSONデータをVolumeにアップロード
# MAGIC
# MAGIC **前提**: このノートブックと同じディレクトリに以下のJSONファイルが配置されていること
# MAGIC - `faq.json` / `customers.json` / `vehicles.json` / `service_cases.json` / `eval_dataset.json`

# COMMAND ----------

context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
notebook_dir = os.path.dirname(context.notebookPath().get())

for filename in ["faq.json", "customers.json", "vehicles.json", "service_cases.json", "eval_dataset.json"]:
    src = f"/Workspace{notebook_dir}/{filename}"
    dst = f"{VOLUME_PATH}/{filename}"
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"✅ Copied: {filename}")
    else:
        print(f"⚠️  Not found: {src}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Bronzeテーブル作成（生データ格納）

# COMMAND ----------

# FAQ ブロンズテーブル
faq_raw = json.load(open(f"{VOLUME_PATH}/faq.json", encoding="utf-8"))
(spark.createDataFrame(faq_raw)
    .write.format("delta").mode("overwrite")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.faq_bronze"))
print(f"✅ FAQ Bronze: {CATALOG}.{SCHEMA}.faq_bronze ({len(faq_raw)}件)")

# COMMAND ----------

# サービス事例 ブロンズテーブル
cases_raw = json.load(open(f"{VOLUME_PATH}/service_cases.json", encoding="utf-8"))
(spark.createDataFrame(cases_raw)
    .write.format("delta").mode("overwrite")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.service_cases_bronze"))
print(f"✅ サービス事例 Bronze ({len(cases_raw)}件)")

# COMMAND ----------

# 顧客マスター
customers_raw = json.load(open(f"{VOLUME_PATH}/customers.json", encoding="utf-8"))
(spark.createDataFrame(customers_raw)
    .write.format("delta").mode("overwrite")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(CUSTOMER_TABLE))
print(f"✅ 顧客マスター: {CUSTOMER_TABLE} ({len(customers_raw)}件)")

# COMMAND ----------

# 車両マスター
vehicles_raw = json.load(open(f"{VOLUME_PATH}/vehicles.json", encoding="utf-8"))
(spark.createDataFrame(vehicles_raw)
    .write.format("delta").mode("overwrite")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(VEHICLE_TABLE))
print(f"✅ 車両マスター: {VEHICLE_TABLE} ({len(vehicles_raw)}件)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Silverテーブル作成（Vector Search用テキスト列を追加）

# COMMAND ----------

# FAQ Silverテーブル（質問+回答を結合した text 列を追加）
df_faq_silver = (
    spark.table(f"{CATALOG}.{SCHEMA}.faq_bronze")
    .withColumn(
        "text",
        F.concat_ws("\n",
            F.concat(F.lit("カテゴリ: "), F.col("category")),
            F.concat(F.lit("質問: "),     F.col("question")),
            F.concat(F.lit("回答: "),     F.col("answer")),
        )
    )
)
(df_faq_silver
    .write.format("delta").mode("overwrite")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(FAQ_TABLE))
print(f"✅ FAQ Silver: {FAQ_TABLE}")
display(df_faq_silver.select("id", "category", "question", "text").limit(2))

# COMMAND ----------

# サービス事例 Silverテーブル（text 列はデータ側に含まれているためそのまま使用）
(spark.table(f"{CATALOG}.{SCHEMA}.service_cases_bronze")
    .write.format("delta").mode("overwrite")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(CASE_TABLE))
print(f"✅ サービス事例 Silver: {CASE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Vector Search エンドポイント作成

# COMMAND ----------

existing = [ep.name for ep in w.vector_search_endpoints.list_endpoints()]
if VS_ENDPOINT_NAME not in existing:
    print(f"Vector Search エンドポイント '{VS_ENDPOINT_NAME}' を作成中...")
    w.vector_search_endpoints.create_endpoint_and_wait(
        name=VS_ENDPOINT_NAME,
        endpoint_type=EndpointType.STANDARD
    )
    print("✅ 作成完了")
else:
    print(f"✅ '{VS_ENDPOINT_NAME}' は既に存在します")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Vector Search インデックス作成

# COMMAND ----------

def create_or_sync_index(index_name, source_table, primary_key, embedding_col):
    existing = [i.name for i in w.vector_search_indexes.list_indexes(endpoint_name=VS_ENDPOINT_NAME)]
    if index_name not in existing:
        print(f"インデックス '{index_name}' を作成中...")
        w.vector_search_indexes.create_index(
            name=index_name,
            endpoint_name=VS_ENDPOINT_NAME,
            primary_key=primary_key,
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=source_table,
                embedding_source_columns=[
                    EmbeddingSourceColumn(name=embedding_col, embedding_model_endpoint_name=EMBEDDING_MODEL)
                ],
                pipeline_type=PipelineType.TRIGGERED
            )
        )
        time.sleep(30)
        for _ in range(20):
            status = w.vector_search_indexes.get_index(index_name=index_name)
            if status.status and status.status.ready:
                print(f"✅ '{index_name}' 同期完了")
                break
            print(f"  同期中... ({status.status.indexed_row_count if status.status else 0} 件)")
            time.sleep(15)
    else:
        print(f"✅ '{index_name}' 既存 → 同期実行")
        w.vector_search_indexes.sync_index(index_name=index_name)

create_or_sync_index(FAQ_INDEX_NAME,  FAQ_TABLE,  "id",      "text")
create_or_sync_index(CASE_INDEX_NAME, CASE_TABLE, "case_id", "text")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Online Table 作成（顧客・車両情報のリアルタイム検索用）

# COMMAND ----------

def create_online_table(table_name, pk_columns):
    online_name = f"{table_name}_online"
    try:
        w.online_tables.get(name=online_name)
        print(f"✅ '{online_name}' は既に存在します")
        return
    except Exception:
        pass
    print(f"'{online_name}' を作成中...")
    w.online_tables.create(
        name=online_name,
        spec=OnlineTableSpec(
            source_table_full_name=table_name,
            primary_key_columns=pk_columns,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy()
        )
    )
    print(f"✅ '{online_name}' 作成完了")

create_online_table(CUSTOMER_TABLE, ["customer_id"])
create_online_table(VEHICLE_TABLE,  ["vehicle_id"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. 動作確認

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vs_client = VectorSearchClient(disable_notice=True)

print("=== FAQ Vector Search テスト ===")
results = vs_client.get_index(
    endpoint_name=VS_ENDPOINT_NAME, index_name=FAQ_INDEX_NAME
).similarity_search(
    query_text="エンジンオイルの交換はいつすればよいですか？",
    columns=["id", "category", "question", "answer"],
    num_results=3
)
for row in results.get("result", {}).get("data_array", []):
    print(f"  [{row[0]}] {row[2][:60]}...")

# COMMAND ----------

print("=== サービス事例 Vector Search テスト ===")
results = vs_client.get_index(
    endpoint_name=VS_ENDPOINT_NAME, index_name=CASE_INDEX_NAME
).similarity_search(
    query_text="バッテリーが上がってしまいました",
    columns=["case_id", "title", "solution"],
    num_results=3
)
for row in results.get("result", {}).get("data_array", []):
    print(f"  [{row[0]}] {row[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## セットアップ完了 ✅
# MAGIC
# MAGIC | リソース | 名前 |
# MAGIC |---------|------|
# MAGIC | Vector Search エンドポイント | `auto_service_vs_endpoint` |
# MAGIC | FAQ インデックス | `faq_vs_index` |
# MAGIC | サービス事例インデックス | `service_cases_vs_index` |
# MAGIC | 顧客 Online Table | `customers_online` |
# MAGIC | 車両 Online Table | `vehicles_online` |
# MAGIC
# MAGIC 次: **02-Agent-Definition** を実行してください。

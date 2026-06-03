# Databricks notebook source
# =============================================================================
# AutoPremium Japan カスタマーサービスエージェント - 設定ファイル
# =============================================================================
# ノートブックは "%run ./config" でこのファイルを読み込みます。
# 環境に合わせてこのファイルのみを変更してください。

# -----------------------------------------------------------------------
# Unity Catalog 設定  ← 環境に合わせて変更してください
# -----------------------------------------------------------------------
CATALOG = "auto_service_demo"   # デプロイ先カタログ名（例: "main", "twatanabe"）
SCHEMA  = "auto_service_demo"   # スキーマ名
VOLUME  = "raw_data"            # データ格納用ボリューム名

# -----------------------------------------------------------------------
# モデル設定
# -----------------------------------------------------------------------
LLM_MODEL       = "databricks-claude-opus-4-7"
EMBEDDING_MODEL = "databricks-gte-large-en"
# 日本語特化の埋め込みモデルを使う場合は以下に変更:
# EMBEDDING_MODEL = "multilingual-e5-large-embedding"

# -----------------------------------------------------------------------
# Vector Search 設定
# -----------------------------------------------------------------------
VS_ENDPOINT_NAME = "auto_service_vs_endpoint"

# -----------------------------------------------------------------------
# エージェント・エンドポイント設定
# -----------------------------------------------------------------------
ENDPOINT_NAME   = "auto-service-agent"
EXPERIMENT_NAME = "/Shared/auto_service_agent_experiment"

# -----------------------------------------------------------------------
# 以下は上記の変数から自動生成されます（変更不要）
# -----------------------------------------------------------------------
FAQ_TABLE        = f"{CATALOG}.{SCHEMA}.faq_silver"
FAQ_INDEX_NAME   = f"{CATALOG}.{SCHEMA}.faq_vs_index"
CASE_TABLE       = f"{CATALOG}.{SCHEMA}.service_cases_silver"
CASE_INDEX_NAME  = f"{CATALOG}.{SCHEMA}.service_cases_vs_index"
CUSTOMER_TABLE   = f"{CATALOG}.{SCHEMA}.customers"
VEHICLE_TABLE    = f"{CATALOG}.{SCHEMA}.vehicles"
AGENT_MODEL_NAME = f"{CATALOG}.{SCHEMA}.auto_service_agent"
VOLUME_PATH      = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

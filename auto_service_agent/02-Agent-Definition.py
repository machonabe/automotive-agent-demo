# Databricks notebook source

# MAGIC %md
# MAGIC # AutoPremium Japan カスタマーサービスエージェント
# MAGIC ## ノートブック 02: エージェント定義・MLflowトレーシング
# MAGIC
# MAGIC このノートブックでは以下を実装します:
# MAGIC 1. エージェントが使用するツール関数の定義
# MAGIC 2. OpenAI Tool Calling を使ったエージェントのメインロジック
# MAGIC 3. MLflow トレーシングによる実行ログの記録
# MAGIC 4. ローカルでのエージェント動作確認
# MAGIC
# MAGIC **アーキテクチャ**:
# MAGIC ```
# MAGIC ユーザーからの質問（顧客ID付き）
# MAGIC       ↓
# MAGIC   Auto Service Agent (LLM with Tools)
# MAGIC       ↓ ツール呼び出し
# MAGIC   ├── get_customer_info()        → 顧客ティア・情報を取得
# MAGIC   ├── get_vehicle_info()         → 保有車両の詳細を取得
# MAGIC   ├── search_faq()               → FAQをVector Searchで検索
# MAGIC   ├── search_service_cases()     → 過去事例をVector Searchで検索
# MAGIC   └── book_service_appointment() → サービス予約（ダミー）
# MAGIC       ↓
# MAGIC   パーソナライズされた回答（ティア特典含む）
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install databricks-sdk mlflow databricks-vectorsearch openai "typing_extensions>=4.12.0" --upgrade -q

# COMMAND ----------

dbutils.library.restartPython()

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
from typing import Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI

DATABRICKS_HOST  = spark.conf.get("spark.databricks.workspaceUrl")
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

mlflow.set_experiment(EXPERIMENT_NAME)

print(f"Workspace : https://{DATABRICKS_HOST}")
print(f"Catalog   : {CATALOG}")
print(f"LLM       : {LLM_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. ツール関数の定義

# COMMAND ----------

@mlflow.trace(name="get_customer_info", span_type="tool")
def get_customer_info(customer_id: str) -> dict:
    """顧客IDから顧客情報（ティア・保有車両IDリスト等）を取得する。"""
    try:
        result = spark.table(CUSTOMER_TABLE).filter(f"customer_id = '{customer_id}'").collect()
        if not result:
            return {"error": f"顧客ID '{customer_id}' が見つかりません。"}
        row = result[0].asDict()
        if isinstance(row.get("vehicle_ids"), str):
            row["vehicle_ids"] = json.loads(row["vehicle_ids"])
        return row
    except Exception as e:
        return {"error": f"顧客情報取得エラー: {str(e)}"}

# COMMAND ----------

@mlflow.trace(name="get_vehicle_info", span_type="tool")
def get_vehicle_info(vehicle_id: str) -> dict:
    """車両IDから車両情報（モデル・保証期限・リコール状況等）を取得する。"""
    try:
        result = spark.table(VEHICLE_TABLE).filter(f"vehicle_id = '{vehicle_id}'").collect()
        if not result:
            return {"error": f"車両ID '{vehicle_id}' が見つかりません。"}
        return result[0].asDict()
    except Exception as e:
        return {"error": f"車両情報取得エラー: {str(e)}"}

# COMMAND ----------

@mlflow.trace(name="search_faq", span_type="retriever")
def search_faq(query: str, customer_tier: str = "standard", num_results: int = 3) -> list:
    """FAQをVector Searchで検索し、顧客ティアに適した回答を返す。"""
    vs_client = VectorSearchClient(disable_notice=True)
    results = vs_client.get_index(
        endpoint_name=VS_ENDPOINT_NAME, index_name=FAQ_INDEX_NAME
    ).similarity_search(
        query_text=query,
        columns=["id", "category", "question", "answer", "applicable_tier"],
        num_results=num_results * 2
    )
    docs = []
    for row in results.get("result", {}).get("data_array", []):
        tier = row[4]
        if tier == "all" or customer_tier in tier:
            docs.append({"id": row[0], "category": row[1], "question": row[2], "answer": row[3], "applicable_tier": tier})
        if len(docs) >= num_results:
            break
    return docs

# COMMAND ----------

@mlflow.trace(name="search_service_cases", span_type="retriever")
def search_service_cases(query: str, num_results: int = 3) -> list:
    """過去のサービス事例をVector Searchで検索する。"""
    vs_client = VectorSearchClient(disable_notice=True)
    results = vs_client.get_index(
        endpoint_name=VS_ENDPOINT_NAME, index_name=CASE_INDEX_NAME
    ).similarity_search(
        query_text=query,
        columns=["case_id", "title", "symptoms", "solution", "cost_estimate", "resolution_time"],
        num_results=num_results
    )
    return [
        {"case_id": r[0], "title": r[1], "symptoms": r[2],
         "solution": r[3], "cost_estimate": r[4], "resolution_time": r[5]}
        for r in results.get("result", {}).get("data_array", [])
    ]

# COMMAND ----------

@mlflow.trace(name="book_service_appointment", span_type="tool")
def book_service_appointment(customer_id: str, vehicle_id: str, service_type: str,
                              preferred_date: str, notes: str = "") -> dict:
    """サービス予約を受け付ける（デモ用ダミー）。"""
    import random
    booking_id = f"SV{random.randint(10000, 99999)}"
    return {
        "booking_id": booking_id, "status": "confirmed",
        "message": f"予約を受け付けました。予約番号: {booking_id}",
        "customer_id": customer_id, "vehicle_id": vehicle_id,
        "service_type": service_type, "scheduled_date": preferred_date, "notes": notes,
        "reminder": "前日にSMSまたはメールにてリマインドをお送りします"
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. ツール定義（OpenAI Tool Calling形式）

# COMMAND ----------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_info",
            "description": "顧客IDから顧客情報（名前・会員ティア・保有車両IDリスト・入会日等）を取得します。ユーザーが顧客IDを提示した場合は必ずこのツールを呼び出してください。",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string", "description": "顧客ID（例: 'C001'）"}},
                "required": ["customer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_vehicle_info",
            "description": "車両IDから車両情報（モデル・年式・保証期限・リコール状況・最終点検日等）を取得します。",
            "parameters": {
                "type": "object",
                "properties": {"vehicle_id": {"type": "string", "description": "車両ID（例: 'V001'）"}},
                "required": ["vehicle_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": "FAQデータベースをベクター検索して関連FAQを取得します。メンテナンス・保証・ロードサービス・会員特典等の質問に使用してください。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "検索クエリ"},
                    "customer_tier": {"type": "string", "enum": ["standard", "silver", "gold"], "description": "顧客の会員ティア"},
                    "num_results": {"type": "integer", "description": "取得するFAQ件数（デフォルト: 3）", "default": 3}
                },
                "required": ["query", "customer_tier"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_service_cases",
            "description": "過去のサービス事例をベクター検索して、症状・問題に類似した事例と解決策を取得します。車の不具合・故障・異音等の質問に使用してください。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "症状・問題の説明"},
                    "num_results": {"type": "integer", "description": "取得する事例件数（デフォルト: 3）", "default": 3}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_service_appointment",
            "description": "サービス予約（定期点検・車検・修理等）を受け付けます。",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "vehicle_id": {"type": "string"},
                    "service_type": {"type": "string", "description": "例: '定期点検', '車検', 'エアコン修理'"},
                    "preferred_date": {"type": "string", "description": "希望日 (YYYY/MM/DD)"},
                    "notes": {"type": "string", "description": "備考・症状の詳細（省略可）"}
                },
                "required": ["customer_id", "vehicle_id", "service_type", "preferred_date"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "get_customer_info": get_customer_info,
    "get_vehicle_info": get_vehicle_info,
    "search_faq": search_faq,
    "search_service_cases": search_service_cases,
    "book_service_appointment": book_service_appointment
}

print(f"✅ {len(TOOLS)} 個のツールを定義しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. システムプロンプト

# COMMAND ----------

SYSTEM_PROMPT = """あなたはAutoPremium Japan（高級自動車ブランド）のカスタマーサービスエージェントです。

## あなたの役割
- お客様の車に関する質問・問題を丁寧かつ正確にサポートする
- 会員ティア（スタンダード/シルバー/ゴールド）に応じた適切な特典・サービスを案内する
- 専門的な技術情報も分かりやすく説明する

## 重要なルール
1. **顧客IDが提示されたら必ず get_customer_info を呼び出す** - ティア・保有車両を確認してからパーソナライズされた回答を提供すること
2. **FAQやサービス事例のツールを積極的に使う** - 推測ではなく、データベースから正確な情報を取得して回答すること
3. **ティア特典を正確に案内する** - スタンダード/シルバー/ゴールド各ティアの特典を混同しないこと
4. **安全に関わる問題は即座に対応** - ブレーキ・タイヤ等の安全問題は緊急性を伝え、速やかにサービス予約を勧める
5. **日本語で丁寧に回答する** - 敬語を使い、親切で分かりやすい説明を心がける

## 提供できないサービス
- 事故の相手方との交渉 / 保険金支払いの確定 / 正式な見積もり（概算のみ提供可）
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. エージェントクラス定義（MLflow ChatModel）

# COMMAND ----------

class AutoServiceAgent(mlflow.pyfunc.ChatModel):
    """AutoPremium Japan カスタマーサービスエージェント"""

    @mlflow.trace(name="auto_service_agent", span_type="chain")
    def predict(self, context, messages, params=None):
        from mlflow.types.llm import ChatCompletionResponse, ChatMessage, ChatChoice

        host  = os.environ.get("DATABRICKS_HOST", f"https://{DATABRICKS_HOST}")
        token = os.environ.get("DATABRICKS_TOKEN", DATABRICKS_TOKEN)
        client = OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")

        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in messages:
            conversation.append({"role": msg.role, "content": msg.content} if hasattr(msg, "role") else msg)

        message = None
        for _ in range(5):  # 最大5回のツール呼び出し
            with mlflow.trace(name="llm_call", span_type="llm"):
                response = client.chat.completions.create(
                    model=LLM_MODEL, messages=conversation,
                    tools=TOOLS, tool_choice="auto",
                    temperature=0.0, max_tokens=2048
                )
            message = response.choices[0].message
            if not message.tool_calls:
                break
            conversation.append(message)
            for tc in message.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                with mlflow.trace(name=f"tool_{func_name}", span_type="tool") as span:
                    span.set_attribute("function_name", func_name)
                    result = TOOL_FUNCTIONS[func_name](**func_args) if func_name in TOOL_FUNCTIONS else {"error": f"未定義: {func_name}"}
                conversation.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False, default=str)})

        final = message.content if message and message.content else "申し訳ありません。回答を生成できませんでした。"
        return ChatCompletionResponse(choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=final))])

print("✅ AutoServiceAgent クラス定義完了")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. ローカルテスト（MLflowトレーシング有効）

# COMMAND ----------

mlflow.autolog(disable=True)
agent = AutoServiceAgent()

from mlflow.types.llm import ChatMessage

test_cases = [
    ("agent_test_gold_member",     "顧客ID C001です。私の車のエンジンオイル交換の時期と費用を教えてください。"),
    ("agent_test_fault_diagnosis", "顧客ID C002です。最近、エンジンをかけるとカラカラという音がします。これは何か問題がありますか？"),
    ("agent_test_safety_urgent",   "顧客ID C003です。ブレーキを踏むとペダルが奥まで沈んでしまいます。大丈夫でしょうか？"),
    ("agent_test_recall_check",    "顧客ID C005です。V006の車がリコール対象と聞きましたが、費用はかかりますか？"),
]

for run_name, question in test_cases:
    print(f"\n{'='*60}\n{run_name}\nQ: {question}\n{'='*60}")
    with mlflow.start_run(run_name=run_name):
        response = agent.predict(context=None, messages=[ChatMessage(role="user", content=question)])
        print(f"A: {response.choices[0].message.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. agent.py の書き出し（デプロイ用）

# COMMAND ----------

context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
notebook_dir = os.path.dirname(context.notebookPath().get())
agent_file_path = f"/Workspace{notebook_dir}/agent.py"

agent_code = f'''# AutoPremium Japan カスタマーサービスエージェント（Serving用）
import json, os
import mlflow
from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI

# --- 設定値（config.py から埋め込み済み） ---
CATALOG          = "{CATALOG}"
SCHEMA           = "{SCHEMA}"
LLM_MODEL        = "{LLM_MODEL}"
VS_ENDPOINT_NAME = "{VS_ENDPOINT_NAME}"
FAQ_INDEX_NAME   = "{FAQ_INDEX_NAME}"
CASE_INDEX_NAME  = "{CASE_INDEX_NAME}"
CUSTOMER_TABLE   = "{CUSTOMER_TABLE}"
VEHICLE_TABLE    = "{VEHICLE_TABLE}"

@mlflow.trace(name="get_customer_info", span_type="tool")
def get_customer_info(customer_id):
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    r = spark.table(CUSTOMER_TABLE).filter(f"customer_id = \\'{{customer_id}}\\'").collect()
    if not r: return {{"error": f"顧客ID \\'{{customer_id}}\\' が見つかりません。"}}
    row = r[0].asDict()
    if isinstance(row.get("vehicle_ids"), str): row["vehicle_ids"] = json.loads(row["vehicle_ids"])
    return row

@mlflow.trace(name="get_vehicle_info", span_type="tool")
def get_vehicle_info(vehicle_id):
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    r = spark.table(VEHICLE_TABLE).filter(f"vehicle_id = \\'{{vehicle_id}}\\'").collect()
    if not r: return {{"error": f"車両ID \\'{{vehicle_id}}\\' が見つかりません。"}}
    return r[0].asDict()

@mlflow.trace(name="search_faq", span_type="retriever")
def search_faq(query, customer_tier="standard", num_results=3):
    vs = VectorSearchClient(disable_notice=True)
    res = vs.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=FAQ_INDEX_NAME).similarity_search(
        query_text=query, columns=["id","category","question","answer","applicable_tier"], num_results=num_results*2)
    docs = []
    for r in res.get("result",{{}}).get("data_array",[]):
        if r[4]=="all" or customer_tier in r[4]:
            docs.append({{"id":r[0],"category":r[1],"question":r[2],"answer":r[3],"applicable_tier":r[4]}})
        if len(docs)>=num_results: break
    return docs

@mlflow.trace(name="search_service_cases", span_type="retriever")
def search_service_cases(query, num_results=3):
    vs = VectorSearchClient(disable_notice=True)
    res = vs.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=CASE_INDEX_NAME).similarity_search(
        query_text=query, columns=["case_id","title","symptoms","solution","cost_estimate","resolution_time"], num_results=num_results)
    return [{{"case_id":r[0],"title":r[1],"symptoms":r[2],"solution":r[3],"cost_estimate":r[4],"resolution_time":r[5]}} for r in res.get("result",{{}}).get("data_array",[])]

@mlflow.trace(name="book_service_appointment", span_type="tool")
def book_service_appointment(customer_id, vehicle_id, service_type, preferred_date, notes=""):
    import random
    bid = f"SV{{random.randint(10000,99999)}}"
    return {{"booking_id":bid,"status":"confirmed","message":f"予約番号: {{bid}}","customer_id":customer_id,"vehicle_id":vehicle_id,"service_type":service_type,"scheduled_date":preferred_date,"notes":notes}}

TOOLS = {json.dumps(TOOLS, ensure_ascii=False)}
TOOL_FUNCTIONS = {{"get_customer_info":get_customer_info,"get_vehicle_info":get_vehicle_info,"search_faq":search_faq,"search_service_cases":search_service_cases,"book_service_appointment":book_service_appointment}}
SYSTEM_PROMPT = """{SYSTEM_PROMPT}"""

class AutoServiceAgent(mlflow.pyfunc.ChatModel):
    @mlflow.trace(name="auto_service_agent", span_type="chain")
    def predict(self, context, messages, params=None):
        from mlflow.types.llm import ChatCompletionResponse, ChatMessage, ChatChoice
        host  = os.environ.get("DATABRICKS_HOST","")
        token = os.environ.get("DATABRICKS_TOKEN","")
        client = OpenAI(api_key=token, base_url=f"{{host}}/serving-endpoints")
        conv = [{{"role":"system","content":SYSTEM_PROMPT}}]
        for m in messages:
            conv.append({{"role":m.role,"content":m.content}} if hasattr(m,"role") else m)
        message = None
        for _ in range(5):
            resp = client.chat.completions.create(model=LLM_MODEL, messages=conv, tools=TOOLS, tool_choice="auto", temperature=0.0, max_tokens=2048)
            message = resp.choices[0].message
            if not message.tool_calls: break
            conv.append(message)
            for tc in message.tool_calls:
                fn, fa = tc.function.name, json.loads(tc.function.arguments)
                result = TOOL_FUNCTIONS[fn](**fa) if fn in TOOL_FUNCTIONS else {{"error":f"未定義: {{fn}}"}}
                conv.append({{"role":"tool","tool_call_id":tc.id,"content":json.dumps(result,ensure_ascii=False,default=str)}})
        final = message.content if message and message.content else "申し訳ありません。"
        return ChatCompletionResponse(choices=[ChatChoice(index=0, message=ChatMessage(role="assistant",content=final))])

mlflow.models.set_model(AutoServiceAgent())
'''

with open(agent_file_path, "w", encoding="utf-8") as f:
    f.write(agent_code)
print(f"✅ agent.py を書き出しました: {agent_file_path}")
print(f"\n次: 03-Deploy-and-Eval-Agent を実行してください。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 完了 ✅
# MAGIC MLflow UI で `{EXPERIMENT_NAME}` を開くとトレースが確認できます。

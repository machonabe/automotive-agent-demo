# AutoPremium Japan カスタマーサービスエージェント デモ

自動車業界向け Databricks Agent デモプロジェクトです。  
顧客ティア（スタンダード/シルバー/ゴールド）に応じたパーソナライズ回答を提供するエージェントを、MLflow による評価・モニタリングとセットで実演します。

---

## アーキテクチャ概要

```
顧客からの問い合わせ（顧客ID付き）
        ↓
  AutoService Agent (LLM with Tools)
        ↓ ツール呼び出し
  ├── get_customer_info()        → ティア・保有車両を確認
  ├── get_vehicle_info()         → 保証期限・リコール状況を確認
  ├── search_faq()               → FAQ を Vector Search
  ├── search_service_cases()     → 過去サービス事例を Vector Search
  └── book_service_appointment() → サービス予約（ダミー）
        ↓
  パーソナライズされた回答（ティア特典・車両固有情報含む）
```

---

## ファイル構成

| ファイル | 説明 |
|----------|------|
| `config.py` | カタログ・モデル名等の設定 |
| `faq.json` | 自動車FAQ データ（25件） |
| `customers.json` | 顧客マスターデータ（8件） |
| `vehicles.json` | 車両マスターデータ（10件） |
| `service_cases.json` | サービス事例データ（15件） |
| `eval_dataset.json` | LLM as Judge 評価用データセット（10件） |
| `01-Data-and-Model-Preparation.py` | データ準備・Vector Search 設定 |
| `02-Agent-Definition.py` | エージェント定義・MLflow トレーシング |
| `03-Deploy-and-Eval-Agent.py` | デプロイ・評価・レビューアプリ |

---

## セットアップ手順

### 前提条件

- Databricks Runtime 15.0+ （MLflow 2.12+ 含む）
- Unity Catalog が有効なワークスペース
- 以下のモデルへのアクセス権:
  - `databricks-claude-opus-4-7`（LLM）
  - `databricks-gte-large-en`（埋め込みモデル）

### 実行順序

1. **このリポジトリをDatabricksワークスペースにインポート**
   - Repos 機能を使ってGitリポジトリとして追加、または
   - ファイルを `auto_service_agent/` ディレクトリごとワークスペースにアップロード

2. **config.py を環境に合わせて修正**（必要な場合）
   ```python
   CATALOG = "main"          # 使用するカタログ名
   SCHEMA  = "auto_service_demo"  # スキーマ名
   ```

3. **01-Data-and-Model-Preparation.py を実行**
   - Unity Catalog セットアップ
   - Delta Tables（FAQ・顧客・車両・サービス事例）作成
   - Vector Search エンドポイント・インデックス作成
   - Online Tables 作成

4. **02-Agent-Definition.py を実行**
   - ツール関数の定義・テスト
   - MLflow トレーシングの動作確認
   - `agent.py` ファイルの生成

5. **03-Deploy-and-Eval-Agent.py を実行**
   - MLflow へのモデル登録
   - Unity Catalog へのモデル登録
   - `databricks.agents.deploy()` でエンドポイント + レビューアプリ作成
   - LLM as Judge 評価の実行

---

## デモシナリオ

### シナリオ1: ゴールド会員の特典案内
```
顧客ID: C001 (田中太郎, ゴールド会員, APX-350 Premium + APX-SUV 450 保有)
質問: 私の車のオイル交換と車検の割引について教えてください
確認点: ゴールド会員の20%割引・無料定期点検が正確に案内されるか
```

### シナリオ2: 故障診断（サービス事例 Vector Search）
```
顧客ID: C002 (鈴木花子, シルバー会員)
質問: エンジンからカラカラ音がします。どうすればよいですか？
確認点: SC010のオイル不足によるタペット音事例が検索・案内されるか
```

### シナリオ3: 安全緊急対応
```
顧客ID: C007 (中村浩介, ゴールド会員)
質問: ブレーキペダルが奥まで沈みます。走行しても大丈夫ですか？
確認点: 緊急性を強調し即座のサービス予約を勧めるか（ゴールド優先対応も案内）
```

### シナリオ4: EV充電トラブル
```
顧客ID: C003 (山田一郎, スタンダード会員, APE-100 Electric)
質問: 急速充電できないエラーが出ます
確認点: SC002のECUソフトウェアアップデート（無償）が案内されるか
```

### シナリオ5: リコール対応確認
```
顧客ID: C005 (伊藤健二, シルバー会員, APV-300 Minivan)
質問: V006のリコールは費用がかかりますか？
確認点: リコール対応費用なし・改修済みステータスが正確に返るか
```

---

## MLflow 評価指標（LLM as Judge）

| 指標 | 説明 | 使用ライブラリ |
|------|------|---------------|
| `answer_correctness` | 期待される回答との正確性 | MLflow 組み込み |
| `answer_relevance` | 質問への回答の関連性 | MLflow 組み込み |
| `faithfulness` | 根拠情報との整合性（幻覚検出） | MLflow 組み込み |
| `customer_tier_compliance` | ティア適合性（カスタム） | `make_genai_metric` |
| `safety_urgency_detection` | 安全緊急性の認識（カスタム） | `make_genai_metric` |

---

## 顧客ティア特典早見表

| 特典 | スタンダード | シルバー | ゴールド |
|------|-------------|---------|---------|
| ロードサービス | 3年間・50km | 保有期間・100km | 保有期間・無制限・海外対応 |
| 整備割引 | なし | 15% | 20% |
| 下取り査定ボーナス | なし | +5% | +10% |
| 車検対応 | 標準 | 15%割引 | 20%割引 + 代車無料 |
| 延長保証割引 | なし | 10% | 20% |
| 優先予約 | なし | なし | 最短翌営業日 |
| 専任担当者 | なし | なし | あり |

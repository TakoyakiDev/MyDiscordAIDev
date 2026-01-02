# Discord RAG Bot (LangGraph強化版)

「ホワイトアウトサバイバル」専門のDiscord RAGボット - LangGraphによる高度なワークフロー管理

## 🎯 概要

このボットは、LangGraphを活用した高度なRAG（Retrieval-Augmented Generation）システムを実装しており、以下の情報源から知識を統合して質問に回答します：

- **Web検索**: Google Custom Search API経由でリアルタイム情報を取得
- **スプレッドシート**: Google Sheetsからセマンティック検索で関連情報を取得
- **会話履歴**: ユーザーごとの会話コンテキストを保持

## ✨ 主要機能

### 🧠 LangGraph強化版の特徴

1. **動的ルーティング**
   - 質問内容に応じて最適な情報源を自動選択
   - キーワードベースのインテリジェントルーティング

2. **並列検索実行**
   - Web検索とスプレッドシート検索を同時実行
   - 処理時間の大幅な短縮

3. **品質スコアリング**
   - コンテキストの品質を0-100点で評価
   - スコアに応じた処理フロー制御

4. **自動リトライ機構**
   - 回答品質が低い場合に自動的に再実行
   - 最大2回までのリトライ

5. **会話履歴管理**
   - ユーザーごとの会話履歴を保持（最新20件）
   - コンテキストを考慮した回答生成

6. **詳細なメトリクス**
   - 各ノードの実行時間を計測
   - 総実行時間と品質スコアを表示

## 🏗️ アーキテクチャ

```
START
  ↓
[初期化] - 状態の初期化
  ↓
[質問分析] - 動的ルーティング判定
  ↓
  ├→ [Web検索]        ⎤ 並列実行
  └→ [Sheet検索]      ⎦
       ↓
  [コンテキスト評価] - 品質スコアリング (0-100点)
       ↓
  ┌────┴────┐
  │ スコア >= 30?
  ├─ YES → [回答生成]
  │          ↓
  │     ┌───┴───┐
  │     │品質OK? │
  │     ├─ NO → [リトライ] ──┐
  │     │         (最大2回)   │
  │     └─ YES               │
  │          ↓               │
  └─ NO → [コンテキスト不足] │
            ↓               │
       [最終処理] ←──────────┘
       - 会話履歴保存
       - メトリクス計算
            ↓
          END
```

## 📦 技術スタック

| カテゴリ | 技術 |
|---------|------|
| **ワークフロー** | LangGraph (状態グラフ管理) |
| **LLM/Embedding** | Ollama (ローカル実行) |
| **LangChain** | langchain-ollama, langchain-core |
| **Discord統合** | discord.py |
| **Web検索** | Google Custom Search API |
| **データソース** | Google Sheets (gspread) |
| **HTML解析** | BeautifulSoup4 |
| **非同期処理** | asyncio, aiohttp |

## 🚀 セットアップ

### 1. 必要な環境

- Python 3.9+
- Ollama (ローカルLLMサーバー)
- Discord Bot Token
- Google Custom Search API
- Google Service Account (Sheets API用)

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 3. Ollamaのセットアップ

```bash
# Ollamaのインストール
curl -fsSL https://ollama.ai/install.sh | sh

# モデルのダウンロード
ollama pull gemma2:2b
ollama pull nomic-embed-text

# Ollamaサーバーの起動
ollama serve
```

### 4. 環境変数の設定

`.env` ファイルを作成し、以下を設定：

```env
# Discord
DISCORD_TOKEN=your_discord_bot_token

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:2b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Google Custom Search
CSE_API_KEY=your_cse_api_key
CSE_CX=your_cse_cx

# Google Sheets
SHEET_ID=your_google_sheet_id
SHEET_NAME=Sheet1
GOOGLE_SERVICE_ACCOUNT_FILE=path/to/service_account.json
```

### 5. ボットの起動

```bash
python discord_rag_bot.py
```

## 🎮 使い方

### 基本コマンド

#### `!ask [質問内容]`
RAGシステムに質問します。

**例:**
```
!ask イベントのおすすめポイントは？
!ask 最新のアップデート情報を教えて
```

**返信形式:**
- 🤖 回答内容
- 📝 質問
- 🔗 参照元（URLリスト）
- ⏱️ 実行時間 | 📊 品質スコア | 🔄 リトライ回数

#### `!history`
会話履歴を表示します（最新5往復）。

#### `!clear`
会話履歴をクリアします。

#### `!graph`
LangGraphのワークフロー構造を可視化します。

#### `!stats`
システム統計情報を表示します。
- 会話ユーザー数
- 総メッセージ数
- 使用モデル
- スプレッドシートデータ行数

#### `!reload` (管理者専用)
スプレッドシートデータを再読み込みします。

## 🔧 カスタマイズ

### LLMモデルの変更

`.env` ファイルで変更可能：

```env
# より高性能なモデルを使用
OLLAMA_MODEL=llama3:8b

# より高精度なEmbeddingモデル
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
```

### 品質スコアの閾値調整

`discord_rag_bot.py` の `evaluate_context` メソッド内で調整：

```python
# 品質スコア計算ロジック
if len(state['combined_context'].strip()) > 100:
    score += 30.0  # この値を調整
```

### リトライ回数の変更

```python
state.setdefault('max_retries', 2)  # デフォルトは2回
```

### コンテキスト長の調整

```python
max_context_length = 2500  # Ollamaのコンテキスト長に応じて調整
```

## 📊 パフォーマンス

### 推定処理時間

| 処理 | 時間 |
|------|------|
| 質問分析 | <0.1秒 |
| Web検索 (並列) | 2-5秒 |
| Sheet検索 (並列) | 1-3秒 |
| コンテキスト評価 | <0.1秒 |
| 回答生成 | 3-10秒 |
| **合計** | **6-18秒** |

※並列実行により従来の30-40%高速化

## 🛡️ エラーハンドリング

- **Web検索失敗**: Sheet検索の結果のみで回答生成
- **Sheet検索失敗**: Web検索の結果のみで回答生成
- **両方失敗**: エラー詳細を含む丁寧なメッセージ表示
- **Ollama接続失敗**: リトライ後、接続エラーメッセージ
- **コンテキスト不足**: 質問の改善提案を表示

## 📝 ファイル構成

```
.
├── discord_rag_bot.py           # メインファイル (LangGraph強化版)
├── discord_rag_bot_langgraph.py # 旧LangGraph版 (参考用)
├── web_search.py                # Web検索Retriever
├── sheets_retriever.py          # Google Sheets Retriever
├── html_parser.py               # HTML解析ユーティリティ
├── requirements.txt             # 依存パッケージ
├── .env                         # 環境変数 (作成必要)
└── README.md                    # このファイル
```

## 🔍 トラブルシューティング

### Ollama接続エラー

```bash
# Ollamaが起動しているか確認
ollama list

# Ollamaサーバーを起動
ollama serve
```

### モデルが見つからない

```bash
# 必要なモデルをダウンロード
ollama pull gemma2:2b
ollama pull nomic-embed-text
```

### Google Sheets接続エラー

1. Service Accountの認証情報が正しいか確認
2. シートIDが正しいか確認
3. Service Accountにシートの閲覧権限があるか確認

### Discord Bot Token エラー

1. Discord Developer Portalでトークンを再生成
2. `.env` ファイルが正しく読み込まれているか確認

## 🚧 制約事項

- **Ollamaのコンテキスト長**: 2500文字に制限
- **Discord埋め込み制限**: 各フィールド1024文字、説明2000文字まで
- **会話履歴**: 最新20件（10往復）のみ保持
- **並列実行**: Web/Sheet検索のみ（LLM呼び出しは逐次）

## 🎯 今後の改善予定

- [ ] Embeddingのキャッシュ機能
- [ ] ストリーミング応答
- [ ] マルチモーダル対応（画像・動画）
- [ ] より高度なクエリ理解（意図分類）
- [ ] A/Bテスト機能
- [ ] グラフ可視化（画像生成）

## 📄 ライセンス

MIT License

## 👨‍💻 開発者

Discord RAG Bot Team

## 🤝 コントリビューション

プルリクエストを歓迎します！

---

**Powered by LangGraph + Ollama + Discord.py**

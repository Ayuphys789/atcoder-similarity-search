# 🔍 AtCoder Similarity Search

AtCoderの競技プログラミング問題に対して、自然言語やキーワードで類似問題を検索できるWebアプリケーションです。
Gemini APIによる要約・ベクトル埋め込みと、Qdrant Cloud上のベクトル検索を組み合わせることで、単なるキーワード一致だけでなく、「意味的に類似した考察が必要な問題」を発見することを目指しました。

**URL**: [https://search-atcoder.ayuphys.com](https://search-atcoder.ayuphys.com)

---

## 🚀 主な機能

### 1. Practice Mode（練習モード）

学習や過去問演習のためのモードです。

- **自然言語検索:** 「最短経路を求める問題」「木DPを使うやつ」といった曖昧なクエリや、問題文のコピペにも対応。
- **AIによる意図解釈:** 入力された自然言語をGemini APIが解析し、アルゴリズム的な文脈（ベクトル）に変換して検索します。

<!-- **実行例:** クエリ「グリッド上の最短経路」
![Practice Mode Screenshot](figs/practice-mode-demo.png) -->

### 2. Contest Mode（コンテストモード）

コンテスト中の使用を想定した、低レイテンシな検索モードです。

- **キーワード検索:** 「DP グラフ」「期待値」などのキーワードで高速にフィルタリング。
- **AI非依存:** Gemini APIを経由せず、QdrantのFull-Text Index機能のみを使用するため、レスポンスが高速です。

<!-- **実行例:** クエリ「ダイクストラ法」
![Contest Mode Screenshot](figs/contest-mode-demo.png) -->

### 3. コンテスト中ロックダウン機能

公平性を担保するため、AtCoderのコンテスト開催時間を自動検知して挙動を制御します。

- **自動検知:** 5分間隔でAtCoder公式のコンテスト情報をポーリング。
- **機能制限:** コンテスト開催中は、AIを使用する `Practice Mode` を自動的に無効化（403 Forbidden）し、`Contest Mode` のみ利用可能にします。

---

## 💻 アーキテクチャ

```
┌───────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Frontend     │────▶│    Backend API   │────▶│  Qdrant Cloud   │
│ (Next.js)     │     │  (Rust / Axum)   │     │  (Vector DB)    │
│  on Vercel    │     │  on Cloud Run    │     │                 │
└───────────────┘     └──────┬───────────┘     └─────────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │  Gemini API  │
                      │  (Google AI) │
                      │  gemini-2.5  │
                      └──────────────┘
```

---

## 🛠 技術スタック

### バックエンド (Rust)

パフォーマンスと型安全性を重視し、Rust (Axum) を採用しました。

| 技術                       | 用途                          |
| :------------------------- | :---------------------------- |
| **Rust** (Edition 2021)    | 言語                          |
| **Axum** 0.8               | Webフレームワーク             |
| **Tokio** 1.0              | 非同期ランタイム              |
| **reqwest** 0.12           | HTTPクライアント              |
| **qdrant-client**          | Vector DBクライアント (gRPC)  |
| **serde** / **serde_json** | シリアライズ / デシリアライズ |
| **tracing**                | 構造化ログ（JSON形式）        |

### フロントエンド (TypeScript)

モダンなReact構成で、Vercel上にデプロイしています。

| 技術               | 用途           |
| :----------------- | :------------- |
| **Next.js** 16     | App Router採用 |
| **React** 19       | UIライブラリ   |
| **TypeScript** 5   | 静的型付け     |
| **Tailwind CSS** 4 | スタイリング   |

### インフラ・データ基盤

Google Cloud Platformを中心としたサーバーレス構成です。

| サービス               | 用途                                                         |
| :--------------------- | :----------------------------------------------------------- |
| **Google Cloud Run**   | バックエンドAPIホスティング                                  |
| **Google Cloud Build** | CI/CD (GitHubへのPushで自動デプロイ)                         |
| **Gemini API**         | `gemini-2.5-flash` (要約), `text-embedding-004` (ベクトル化) |
| **Qdrant Cloud**       | マネージドベクトルデータベース                               |
| **Terraform**          | IaCによるインフラ管理                                        |

---

## 🔌 API仕様

### `GET /health`

サーバーの健全性と、現在の「コンテストロックダウン」状態を返します。

```json
{
  "status": "ok",
  "lockdown": false,
  "running_contest": null
}
```

### `POST /search`

検索クエリを受け取り、類似問題を返します。

**Request:**

```json
{
  "query": "最短経路を求める問題",
  "mode": "practice" // "practice" or "contest"
}
```

**Response Example:**

```json
{
  "results": [
    {
      "problem_id": "abc252_e",
      "title": "Road Reduction",
      "contest_id": "abc252",
      "summary": "ダイクストラ法を用いて最短経路木を構築する...",
      "score": 0.872
    }
  ]
}
```

---

## ⚙️ セットアップと開発

### 1. 前提条件

- Rust (Cargo), Node.js, Python 3.9+, Docker
- Gemini API Key, Qdrant Cloud Cluster

### 2. 環境変数

各ディレクトリ（`backend`, `frontend`, `scripts`）に `.env` ファイルを作成してください。

```bash
# backend/.env
GEMINI_API_KEY=xxx
QDRANT_URL=[https://xxx.qdrant.io](https://xxx.qdrant.io)
QDRANT_API_KEY=xxx
```

### 3. バックエンド起動

```bash
cd backend
cargo run
# -> Server listening on [http://0.0.0.0:8080](http://0.0.0.0:8080)
```

### 4. フロントエンド起動

```bash
cd frontend
npm install && npm run dev
# -> Ready on http://localhost:3000
```

### 5. データインジェスト (Python)

過去問データを取得・ベクトル化し、Qdrantへ登録します。

```bash
cd scripts
pip install -r requirements.txt
python ingest.py --reprocess-recent 10  # 直近10コンテスト分を更新
```

---

## 📂 プロジェクト構成

```
.
├── backend/            # Rust (Axum) API Server
│   ├── src/
│   ├── Cargo.toml
│   └── Dockerfile
├── frontend/           # Next.js App Router
│   ├── src/app/
│   └── package.json
├── scripts/            # Python Data Pipeline
│   └── ingest.py       # Fetch & Vectorize & Upsert
├── terraform/          # IaC for Cloud Run
└── cloudbuild.yaml     # CI/CD Configuration
```

---

## 📜 ライセンス

[MIT License](LICENSE)

## 👋 作成者

三宅 智史 (Satoshi Miyake) - [GitHub: Ayuphys789](https://github.com/Ayuphys789)

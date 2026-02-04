# AtCoder Similarity Search

AtCoderの競技プログラミング問題に対して、自然言語やキーワードで類似問題を検索できるWebアプリケーション。

Gemini APIによるAI要約・ベクトル埋め込みと、Qdrant Cloud上のベクトル検索を組み合わせ、意味的に類似した問題を高精度に発見できる。

**URL**: [https://search-atcoder.ayuphys.com](https://search-atcoder.ayuphys.com)

---

## 機能

### Practice Mode（練習モード）

- 自然言語でクエリを入力（例：「最短経路を求める問題」）
- Gemini APIがクエリを数理的・アルゴリズム的キーワードに変換
- ベクトル埋め込みを生成し、Qdrantでコサイン類似度検索を実行
- 類似度スコア付きで結果を表示

### Contest Mode（コンテストモード）

- キーワードベースの高速検索（例：「DP グラフ」）
- Qdrantのフルテキストインデックスによるキーワードマッチング
- AI処理を介さないため低レイテンシ
- コンテスト中も常時利用可能

### コンテスト中ロックダウン機能

- AtCoderのコンテストスケジュールを5分間隔でポーリング
- コンテスト開催中はPractice Modeを自動的に無効化（403 Forbidden）
- Contest Modeは影響を受けず利用可能

---

## アーキテクチャ

```
┌───────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend    │────▶│    Backend API   │────▶│  Qdrant Cloud   │
│  (Next.js)    │     │  (Rust / Axum)   │     │  (Vector DB)    │
│   on Vercel   │     │  on Cloud Run    │     │                 │
└───────────────┘     └──────┬───────────┘     └─────────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │  Gemini API  │
                      │  (Google AI) │
                      └──────────────┘
```

---

## 技術スタック

### バックエンド

| 技術                                 | 用途                          |
| ------------------------------------ | ----------------------------- |
| **Rust** (Edition 2021)              | 言語                          |
| **Axum** 0.8                         | Webフレームワーク             |
| **Tokio** 1                          | 非同期ランタイム              |
| **reqwest** 0.12                     | HTTPクライアント              |
| **qdrant-client** 1                  | Qdrant gRPCクライアント       |
| **tower-http** 0.6                   | CORSミドルウェア              |
| **tracing** / **tracing-subscriber** | 構造化ログ（JSON形式）        |
| **serde** / **serde_json**           | シリアライズ / デシリアライズ |
| **thiserror** 2                      | エラーハンドリング            |
| **dotenvy** 0.15                     | 環境変数読み込み              |

### フロントエンド

| 技術               | 用途                        |
| ------------------ | --------------------------- |
| **Next.js** 16     | Reactフレームワーク         |
| **React** 19       | UIライブラリ                |
| **TypeScript** 5   | 型安全な開発                |
| **Tailwind CSS** 4 | ユーティリティファーストCSS |

### データパイプライン（Python スクリプト）

| ライブラリ         | 用途                                                  |
| ------------------ | ----------------------------------------------------- |
| **google-genai**   | Gemini API クライアント（要約生成・ベクトル埋め込み） |
| **qdrant-client**  | Qdrant Cloud へのデータアップロード                   |
| **requests**       | AtCoder問題文の取得                                   |
| **BeautifulSoup4** | HTML パース                                           |
| **python-dotenv**  | 環境変数管理                                          |

### インフラ・SaaS

| サービス                                                  | 用途                                             |
| --------------------------------------------------------- | ------------------------------------------------ |
| **Google Cloud Run**                                      | バックエンドAPIのホスティング（asia-northeast1） |
| **Google Cloud Build**                                    | CI/CDパイプライン                                |
| **Google Artifact Registry**                              | Dockerイメージの管理                             |
| **Google Secret Manager**                                 | APIキーの安全な管理                              |
| **Gemini API** (`gemini-2.5-flash`, `text-embedding-004`) | クエリ要約生成・ベクトル埋め込み（768次元）      |
| **Qdrant Cloud**                                          | ベクトルデータベース（コサイン類似度検索）       |
| **Vercel**                                                | フロントエンドのホスティング                     |
| **Cloudflare Registrar**                                  | ドメイン管理・DNS                                |
| **Terraform**                                             | インフラのコード管理（IaC）                      |

---

## APIエンドポイント

### `GET /health`

サーバーの状態とコンテストロックダウン状況を返す。

**レスポンス例:**

```json
{
  "status": "ok",
  "lockdown": false,
  "running_contest": null
}
```

### `POST /search`

問題の類似検索を実行する。

**リクエスト:**

```json
{
  "query": "最短経路を求める問題",
  "mode": "practice"
}
```

**レスポンス例:**

```json
{
  "results": [
    {
      "problem_id": "abc252_e",
      "title": "Road Reduction",
      "contest_id": "abc252",
      "summary": "最小全域木を求める問題。Kruskal法またはPrim法...",
      "score": 0.872
    }
  ]
}
```

| モード     | 処理フロー                                            | AI使用 |
| ---------- | ----------------------------------------------------- | ------ |
| `practice` | クエリ → Gemini要約 → ベクトル埋め込み → ベクトル検索 | あり   |
| `contest`  | クエリ → キーワード分割 → テキストマッチフィルタ      | なし   |

---

## セットアップ

### 前提条件

- Rust (Edition 2021)
- Node.js 18+
- Python 3.9+
- Docker
- Gemini API キー
- Qdrant Cloud アカウント

### 環境変数

**バックエンド** (`.env`):

```
GEMINI_API_KEY=your-gemini-api-key
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6334
QDRANT_API_KEY=your-qdrant-api-key
```

**フロントエンド** (`frontend/.env.local`):

```
NEXT_PUBLIC_API_URL=https://your-cloud-run-url.run.app
```

**データパイプライン** (`scripts/.env`):

```
GEMINI_API_KEY=your-gemini-api-key
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6334
QDRANT_API_KEY=your-qdrant-api-key
```

### バックエンド起動

```bash
cd backend
cargo run
```

サーバーが `http://localhost:8080` で起動する。

### フロントエンド起動

```bash
cd frontend
npm install
npm run dev
```

`http://localhost:3000` で開発サーバーが起動する。

### データインジェスト

全ABCコンテストの問題をQdrant Cloudに登録する。

```bash
cd scripts
pip install -r requirements.txt
python ingest.py
```

**オプション:**

```bash
# 特定のコンテスト以降を再処理
python ingest.py --from-contest abc300

# 最近N件のコンテストを再処理
python ingest.py --reprocess-recent 5
```

インジェストスクリプトはレジューム機能を備えており、中断しても既に処理済みの問題をスキップして再開できる。

---

## デプロイ

### バックエンド（Cloud Run）

Cloud Buildによる自動デプロイ:

```bash
gcloud builds submit --config cloudbuild.yaml
```

または手動でDockerビルド:

```bash
cd backend
docker build -t atcoder-similarity-search-api .
```

### フロントエンド（Vercel）

GitHubリポジトリをVercelに接続し、`frontend` ディレクトリをルートディレクトリとして設定する。環境変数 `NEXT_PUBLIC_API_URL` をVercelのダッシュボードで設定する。

### インフラ（Terraform）

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

---

## プロジェクト構成

```
.
├── backend/
│   ├── src/
│   │   └── main.rs          # APIサーバー本体
│   ├── Cargo.toml            # Rust依存関係
│   ├── Dockerfile            # マルチステージビルド
│   └── .dockerignore
├── frontend/
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx      # 検索UI
│   │       ├── layout.tsx    # ルートレイアウト
│   │       └── globals.css   # グローバルスタイル
│   ├── package.json
│   ├── tsconfig.json
│   └── .env.example
├── scripts/
│   ├── ingest.py             # データインジェストスクリプト
│   └── requirements.txt
├── terraform/
│   ├── main.tf               # Cloud Run定義
│   ├── variables.tf          # 変数定義
│   └── outputs.tf            # 出力定義
├── cloudbuild.yaml           # CI/CDパイプライン
└── README.md
```

---

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 作成者

三宅 智史 (Satoshi Miyake) (Ayuphys789)

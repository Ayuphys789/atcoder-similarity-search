#!/usr/bin/env python3
"""
AtCoder問題データをQdrant Cloudにインジェストするスクリプト。

1. atcoder-problems-apiから直近のABCデータを取得
2. Gemini APIで数理的要約を作成
3. Gemini APIでベクトル化
4. Qdrant Cloudにアップロード
"""

import os
import re
import time
import requests
from dotenv import load_dotenv
from google import genai
from bs4 import BeautifulSoup

# .envファイルを読み込み
load_dotenv()
from google.genai import errors as genai_errors
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, TextIndexParams, TokenizerType

# 環境変数
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

# 定数
COLLECTION_NAME = "atcoder_problems"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIM = 768
GENERATION_MODEL = "gemini-2.5-pro"
ABC_PREFIX = "abc"
PROBLEMS_API = "https://kenkoooo.com/atcoder/resources/problems.json"
CONTESTS_API = "https://kenkoooo.com/atcoder/resources/contests.json"

# Geminiクライアント
client: genai.Client = None


def fetch_recent_abc_problems(limit: int = 1) -> list[dict]:
    """直近のABCコンテストの問題を取得する。"""
    print("Fetching contests...")
    contests = requests.get(CONTESTS_API, timeout=30).json()

    now = int(time.time())

    # ABCコンテストをフィルタ（終了済みのみ）し、開始時刻で降順ソート
    abc_contests = [
        c for c in contests
        if c["id"].startswith(ABC_PREFIX)
        and c["id"][3:].isdigit()
        and c["start_epoch_second"] + c["duration_second"] < now  # 終了済みのみ
    ]
    abc_contests.sort(key=lambda c: c["start_epoch_second"], reverse=True)
    recent_abc_ids = {c["id"] for c in abc_contests[:limit]}

    print(f"Fetching problems for {len(recent_abc_ids)} contests...")
    problems = requests.get(PROBLEMS_API, timeout=30).json()

    # 直近ABCの問題をフィルタ
    recent_problems = [
        p for p in problems
        if p["contest_id"] in recent_abc_ids
    ]

    print(f"Found {len(recent_problems)} problems")
    return recent_problems


def fetch_problem_statement(problem: dict) -> str:
    """AtCoderから問題文を取得する。"""
    url = f"https://atcoder.jp/contests/{problem['contest_id']}/tasks/{problem['id']}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # 問題文セクションを取得（日本語版）
        task_statement = soup.find("div", id="task-statement")
        if not task_statement:
            return ""

        # 日本語部分を優先（lang-jaクラス）
        ja_section = task_statement.find("span", class_="lang-ja")
        if ja_section:
            text = ja_section.get_text(separator="\n", strip=True)
        else:
            text = task_statement.get_text(separator="\n", strip=True)

        # 余分な空白を整理
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 長すぎる場合は切り詰め（Geminiのコンテキスト制限対策）
        if len(text) > 4000:
            text = text[:4000] + "..."

        return text
    except Exception as e:
        print(f"    Warning: Failed to fetch problem statement: {e}")
        return ""


def retry_on_rate_limit(func, *args, max_retries=5, **kwargs):
    """レート制限やサーバーエラー時にリトライする。"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except genai_errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 30 * (attempt + 1)
                print(f"    Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise
        except genai_errors.ServerError as e:
            # 503 UNAVAILABLE (model overloaded) などのサーバーエラー
            wait_time = 60 * (attempt + 1)
            print(f"    Server error ({e}), waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded")


def generate_summary(problem: dict, statement: str) -> str:
    """Gemini APIで問題の数理的要約を生成する。"""
    prompt = f"""以下のAtCoder競技プログラミング問題について、数理的・アルゴリズム的な観点から要約してください。
使用するデータ構造、アルゴリズム、計算量などを簡潔に記述してください。

問題ID: {problem["id"]}
問題名: {problem["title"]}
コンテスト: {problem["contest_id"]}

--- 問題文 ---
{statement if statement else "(問題文取得失敗)"}
--- 問題文終わり ---

要約（日本語、100-200字程度）:"""

    def _call():
        return client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )

    response = retry_on_rate_limit(_call)
    return response.text.strip()


def generate_embedding(text: str) -> list[float]:
    """Gemini APIでテキストをベクトル化する。"""
    def _call():
        return client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text
        )

    response = retry_on_rate_limit(_call)
    return response.embeddings[0].values


def init_qdrant() -> QdrantClient:
    """Qdrantクライアントを初期化し、コレクションを作成する。"""
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # コレクションが存在しなければ作成
    collections = qdrant.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        print(f"Creating collection: {COLLECTION_NAME}")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )

    # summaryフィールドにテキストインデックスを作成（キーワード検索用）
    print("Creating text index on summary field...")
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="summary",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.MULTILINGUAL,
            min_token_len=2,
            max_token_len=20,
        )
    )

    return qdrant


def upsert_problem(qdrant: QdrantClient, problem: dict, summary: str, embedding: list[float]):
    """問題データをQdrantにアップロードする。"""
    point = PointStruct(
        id=hash(problem["id"]) & 0x7FFFFFFFFFFFFFFF,  # 正の整数に変換
        vector=embedding,
        payload={
            "problem_id": problem["id"],
            "title": problem["title"],
            "contest_id": problem["contest_id"],
            "summary": summary,
        }
    )
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])


def main():
    global client

    print("=== AtCoder Problem Ingestion ===\n")

    # Qdrant初期化（先にコレクションを確保）
    qdrant = init_qdrant()

    # Gemini API初期化
    client = genai.Client(api_key=GEMINI_API_KEY)

    # 直近ABCの問題を取得
    problems = fetch_recent_abc_problems(limit=1)

    # 各問題を処理
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] Processing: {problem['id']} - {problem['title']}")

        # 問題文取得
        print("  Fetching problem statement...")
        statement = fetch_problem_statement(problem)
        print(f"  Statement length: {len(statement)} chars")
        time.sleep(1)  # AtCoderへの負荷軽減

        # 要約生成
        print("  Generating summary...")
        summary = generate_summary(problem, statement)
        print(f"  Summary: {summary[:80]}...")

        # ベクトル化
        print("  Generating embedding...")
        embedding = generate_embedding(summary)

        # Qdrantにアップロード
        print("  Uploading to Qdrant...")
        upsert_problem(qdrant, problem, summary, embedding)

        # レート制限対策
        time.sleep(5)

    print(f"\n=== Done! Ingested {len(problems)} problems ===")


if __name__ == "__main__":
    main()

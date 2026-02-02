#!/usr/bin/env python3
"""
AtCoder問題データをQdrant Cloudにインジェストするスクリプト。

1. atcoder-problems-apiから直近のABCデータを取得
2. Gemini APIで数理的要約を作成
3. Gemini APIでベクトル化
4. Qdrant Cloudにアップロード
"""

import os
import time
import requests
from dotenv import load_dotenv
from google import genai

# .envファイルを読み込み
load_dotenv()
from google.genai import errors as genai_errors
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

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


def fetch_recent_abc_problems(limit: int = 10) -> list[dict]:
    """直近のABCコンテストの問題を取得する。"""
    print("Fetching contests...")
    contests = requests.get(CONTESTS_API, timeout=30).json()

    # ABCコンテストをフィルタし、開始時刻で降順ソート
    abc_contests = [
        c for c in contests
        if c["id"].startswith(ABC_PREFIX) and c["id"][3:].isdigit()
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


def generate_summary(problem: dict) -> str:
    """Gemini APIで問題の数理的要約を生成する。"""
    prompt = f"""以下のAtCoder競技プログラミング問題について、数理的・アルゴリズム的な観点から要約してください。
使用するデータ構造、アルゴリズム、計算量などを簡潔に記述してください。

問題ID: {problem["id"]}
問題名: {problem["title"]}
コンテスト: {problem["contest_id"]}

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

    # Gemini API初期化
    client = genai.Client(api_key=GEMINI_API_KEY)

    # 直近ABCの問題を取得
    problems = fetch_recent_abc_problems(limit=10)

    # Qdrant初期化
    qdrant = init_qdrant()

    # 各問題を処理
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] Processing: {problem['id']} - {problem['title']}")

        # 要約生成
        print("  Generating summary...")
        summary = generate_summary(problem)
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

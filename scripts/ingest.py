#!/usr/bin/env python3
"""
AtCoder問題データをQdrant Cloudにインジェストするスクリプト。

1. atcoder-problems-apiから直近のABCデータを取得
2. Gemini APIで数理的要約を作成
3. OpenAI APIでベクトル化
4. Qdrant Cloudにアップロード
"""

import os
import time
import requests
from openai import OpenAI
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 環境変数
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

# 定数
COLLECTION_NAME = "atcoder_problems"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
ABC_PREFIX = "abc"
PROBLEMS_API = "https://kenkoooo.com/atcoder/resources/problems.json"
CONTESTS_API = "https://kenkoooo.com/atcoder/resources/contests.json"


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


def generate_summary(problem: dict) -> str:
    """Gemini APIで問題の数理的要約を生成する。"""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""以下のAtCoder競技プログラミング問題について、数理的・アルゴリズム的な観点から要約してください。
使用するデータ構造、アルゴリズム、計算量などを簡潔に記述してください。

問題ID: {problem["id"]}
問題名: {problem["title"]}
コンテスト: {problem["contest_id"]}

要約（日本語、100-200字程度）:"""

    response = model.generate_content(prompt)
    return response.text.strip()


def generate_embedding(text: str) -> list[float]:
    """OpenAI APIでテキストをベクトル化する。"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def init_qdrant() -> QdrantClient:
    """Qdrantクライアントを初期化し、コレクションを作成する。"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # コレクションが存在しなければ作成
    collections = client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        print(f"Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )

    return client


def upsert_problem(client: QdrantClient, problem: dict, summary: str, embedding: list[float]):
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
    client.upsert(collection_name=COLLECTION_NAME, points=[point])


def main():
    print("=== AtCoder Problem Ingestion ===\n")

    # 直近ABCの問題を取得
    problems = fetch_recent_abc_problems(limit=10)

    # Qdrant初期化
    client = init_qdrant()

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
        upsert_problem(client, problem, summary, embedding)

        # レート制限対策
        time.sleep(1)

    print(f"\n=== Done! Ingested {len(problems)} problems ===")


if __name__ == "__main__":
    main()

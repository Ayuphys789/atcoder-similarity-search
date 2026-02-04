#!/usr/bin/env python3
"""
AtCoder問題データをQdrant Cloudにインジェストするスクリプト。

1. atcoder-problems-apiから直近のABCデータを取得
2. Gemini APIで数理的要約を作成
3. Gemini APIでベクトル化
4. Qdrant Cloudにアップロード
"""

import argparse
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


def fetch_all_abc_problems() -> list[dict]:
    """全てのABCコンテストの問題を取得する（古い順）。"""
    print("Fetching contests...")
    contests = requests.get(CONTESTS_API, timeout=30).json()

    now = int(time.time())

    # ABCコンテストをフィルタ（終了済みのみ）し、開始時刻で昇順ソート（古い順）
    abc_contests = [
        c for c in contests
        if c["id"].startswith(ABC_PREFIX)
        and c["id"][3:].isdigit()
        and c["start_epoch_second"] + c["duration_second"] < now  # 終了済みのみ
    ]
    abc_contests.sort(key=lambda c: c["start_epoch_second"])  # 古い順
    abc_contest_ids = {c["id"] for c in abc_contests}

    print(f"Found {len(abc_contest_ids)} ABC contests")

    print("Fetching problems...")
    problems = requests.get(PROBLEMS_API, timeout=30).json()

    # ABCの問題をフィルタし、contest_idでソート
    abc_problems = [
        p for p in problems
        if p["contest_id"] in abc_contest_ids
    ]
    abc_problems.sort(key=lambda p: (p["contest_id"], p["id"]))

    print(f"Found {len(abc_problems)} problems total")
    return abc_problems


def get_existing_problem_ids(qdrant: QdrantClient) -> set[str]:
    """既にQdrantに登録済みの問題IDを取得する。"""
    existing_ids = set()
    offset = None

    while True:
        result = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=["problem_id"],
            with_vectors=False,
        )
        points, offset = result

        for point in points:
            if point.payload and "problem_id" in point.payload:
                existing_ids.add(point.payload["problem_id"])

        if offset is None:
            break

    return existing_ids


def delete_problems_by_contest(qdrant: QdrantClient, contest_ids: set[str]) -> int:
    """指定されたコンテストIDの問題をQdrantから削除する。"""
    # 全ポイントをスクロールして対象を特定
    point_ids_to_delete = []
    offset = None

    while True:
        result = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=["contest_id"],
            with_vectors=False,
        )
        points, offset = result

        for point in points:
            if point.payload and point.payload.get("contest_id") in contest_ids:
                point_ids_to_delete.append(point.id)

        if offset is None:
            break

    # 削除実行
    if point_ids_to_delete:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=point_ids_to_delete
        )
        print(f"  Deleted {len(point_ids_to_delete)} problems")

    return len(point_ids_to_delete)


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

    parser = argparse.ArgumentParser(description="AtCoder問題データをQdrantにインジェスト")
    parser.add_argument("--from-contest", type=str, help="指定コンテスト以降を再処理 (例: abc008)")
    parser.add_argument("--reprocess-recent", type=int, default=0, help="最近N件のコンテストを再処理")
    args = parser.parse_args()

    print("=== AtCoder Problem Ingestion (Full) ===\n")

    # Qdrant初期化（先にコレクションを確保）
    qdrant = init_qdrant()

    # 全ABCの問題を取得
    problems = fetch_all_abc_problems()

    # コンテストIDリスト（古い順）
    contest_ids_ordered = []
    seen = set()
    for p in problems:
        if p["contest_id"] not in seen:
            contest_ids_ordered.append(p["contest_id"])
            seen.add(p["contest_id"])

    # 削除対象のコンテストを決定
    contests_to_delete = set()

    if args.from_contest:
        # 指定コンテスト以降を削除
        start_idx = None
        for i, cid in enumerate(contest_ids_ordered):
            if cid == args.from_contest or cid.startswith(args.from_contest):
                start_idx = i
                break
        if start_idx is not None:
            contests_to_delete.update(contest_ids_ordered[start_idx:])
            print(f"Will reprocess from {args.from_contest}: {len(contests_to_delete)} contests")

    if args.reprocess_recent > 0:
        # 最近N件のコンテストを削除
        recent = contest_ids_ordered[-args.reprocess_recent:]
        contests_to_delete.update(recent)
        print(f"Will reprocess recent {args.reprocess_recent} contests: {recent}")

    # 削除実行
    if contests_to_delete:
        print(f"\nDeleting {len(contests_to_delete)} contests from Qdrant...")
        delete_problems_by_contest(qdrant, contests_to_delete)
        print("Deletion complete.\n")

    # 既存の問題IDを取得（レジューム用）
    print("Checking existing problems in Qdrant...")
    existing_ids = get_existing_problem_ids(qdrant)
    print(f"Already ingested: {len(existing_ids)} problems")

    # Gemini API初期化
    client = genai.Client(api_key=GEMINI_API_KEY)

    # スキップ済み・処理対象を分離
    problems_to_process = [p for p in problems if p["id"] not in existing_ids]
    print(f"To process: {len(problems_to_process)} problems (skipping {len(existing_ids)} already done)\n")

    if not problems_to_process:
        print("All problems already ingested!")
        return

    # 各問題を処理
    success_count = 0
    error_count = 0

    for i, problem in enumerate(problems_to_process):
        print(f"\n[{i+1}/{len(problems_to_process)}] Processing: {problem['id']} - {problem['title']}")

        try:
            # 問題文取得
            print("  Fetching problem statement...")
            statement = fetch_problem_statement(problem)
            print(f"  Statement length: {len(statement)} chars")
            time.sleep(2)  # AtCoderへの負荷軽減

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

            success_count += 1
            print(f"  ✓ Done (total: {success_count + len(existing_ids)})")

        except Exception as e:
            error_count += 1
            print(f"  ✗ Error: {e}")
            # エラーが続く場合は少し待つ
            time.sleep(30)
            continue

        # レート制限対策（Gemini API: 15 RPM for free tier）
        # 1問あたり2リクエスト（summary + embedding）なので、8秒間隔
        time.sleep(8)

    print(f"\n=== Done! ===")
    print(f"  New: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total in DB: {success_count + len(existing_ids)}")


if __name__ == "__main__":
    main()

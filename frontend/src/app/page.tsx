"use client";

import { useState } from "react";

type SearchMode = "practice" | "contest";

interface SearchResult {
  problem_id: string;
  title: string;
  contest_id: string;
  summary: string;
  score: number;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

export default function Home() {
  const [mode, setMode] = useState<SearchMode>("practice");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query.trim(),
          mode,
          limit: 10,
        }),
      });

      if (!response.ok) {
        if (response.status === 403) {
          throw new Error("コンテスト開催中のため、Practice Modeは使用できません");
        }
        throw new Error(`検索エラー: ${response.status}`);
      }

      const data = await response.json();
      setResults(data.results || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "検索に失敗しました");
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-4xl mx-auto px-4 py-12">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            AtCoder Similarity Search
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            類似問題を検索して練習効率を上げよう
          </p>
        </header>

        {/* Mode Toggle */}
        <div className="flex justify-center mb-6">
          <div className="inline-flex rounded-lg border border-gray-200 dark:border-gray-700 p-1 bg-white dark:bg-gray-800">
            <button
              onClick={() => setMode("practice")}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                mode === "practice"
                  ? "bg-blue-500 text-white"
                  : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
              }`}
            >
              Practice Mode
            </button>
            <button
              onClick={() => setMode("contest")}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                mode === "contest"
                  ? "bg-green-500 text-white"
                  : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
              }`}
            >
              Contest Mode
            </button>
          </div>
        </div>

        {/* Mode Description */}
        <p className="text-center text-sm text-gray-500 dark:text-gray-500 mb-6">
          {mode === "practice"
            ? "AI要約によるベクトル検索（コンテスト中は無効）"
            : "キーワードによる高速検索（常時利用可能）"}
        </p>

        {/* Search Form */}
        <form onSubmit={handleSearch} className="mb-8">
          <div className="flex gap-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                mode === "practice"
                  ? "例: 最短経路を求める問題"
                  : "例: DP グラフ"
              }
              className="flex-1 px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="px-6 py-3 rounded-lg bg-blue-500 text-white font-medium hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? "検索中..." : "検索"}
            </button>
          </div>
        </form>

        {/* Error */}
        {error && (
          <div className="mb-6 p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
            <p className="text-red-600 dark:text-red-400">{error}</p>
          </div>
        )}

        {/* Results */}
        {results.length > 0 && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              検索結果 ({results.length}件)
            </h2>
            {results.map((result, index) => (
              <div
                key={result.problem_id}
                className="p-4 rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600 transition-colors"
              >
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <a
                      href={`https://atcoder.jp/contests/${result.contest_id}/tasks/${result.problem_id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 dark:text-blue-400 hover:underline font-medium"
                    >
                      {result.title}
                    </a>
                    <span className="ml-2 text-sm text-gray-500 dark:text-gray-400">
                      ({result.contest_id})
                    </span>
                  </div>
                  <span className="text-sm text-gray-400 dark:text-gray-500">
                    #{index + 1}
                  </span>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  {result.summary}
                </p>
                {result.score > 0 && (
                  <div className="mt-2 text-xs text-gray-400 dark:text-gray-500">
                    類似度: {(result.score * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Empty State */}
        {!loading && !error && results.length === 0 && query && (
          <div className="text-center py-12 text-gray-500 dark:text-gray-400">
            検索結果がありません
          </div>
        )}

        {/* Initial State */}
        {!loading && !error && results.length === 0 && !query && (
          <div className="text-center py-12 text-gray-400 dark:text-gray-500">
            検索クエリを入力してください
          </div>
        )}
      </div>
    </div>
  );
}

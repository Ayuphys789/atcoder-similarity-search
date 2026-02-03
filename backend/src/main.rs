use axum::{
    extract::State,
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use qdrant_client::{
    qdrant::{
        with_payload_selector::SelectorOptions, Condition, Filter, PayloadIncludeSelector,
        SearchPoints, WithPayloadSelector,
    },
    Qdrant,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

// ============================================================================
// Constants
// ============================================================================

const COLLECTION_NAME: &str = "atcoder_problems";
const EMBEDDING_MODEL: &str = "text-embedding-004";
const GENERATION_MODEL: &str = "gemini-2.5-flash";
const EMBEDDING_DIM: usize = 768;

const CONTESTS_API: &str = "https://kenkoooo.com/atcoder/resources/contests.json";
const SCHEDULE_POLL_INTERVAL_SECS: u64 = 300; // 5 minutes

// ============================================================================
// Contest Schedule Types
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
struct AtCoderContest {
    id: String,
    start_epoch_second: i64,
    duration_second: i64,
}

#[derive(Debug, Clone, Default)]
struct ContestSchedule {
    contests: Vec<AtCoderContest>,
    last_updated: u64,
}

impl ContestSchedule {
    /// Check if any contest is currently running
    fn is_contest_running(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        self.contests.iter().any(|contest| {
            let start = contest.start_epoch_second;
            let end = start + contest.duration_second;
            now >= start && now < end
        })
    }

    /// Get the currently running contest (if any)
    fn get_running_contest(&self) -> Option<&AtCoderContest> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        self.contests.iter().find(|contest| {
            let start = contest.start_epoch_second;
            let end = start + contest.duration_second;
            now >= start && now < end
        })
    }
}

// ============================================================================
// AppState
// ============================================================================

#[derive(Clone)]
struct AppState {
    qdrant: Arc<Qdrant>,
    gemini_api_key: String,
    http_client: reqwest::Client,
    contest_schedule: Arc<RwLock<ContestSchedule>>,
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct SearchRequest {
    query: String,
    mode: SearchMode,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum SearchMode {
    Practice,
    Contest,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
}

#[derive(Debug, Serialize)]
struct SearchResult {
    problem_id: String,
    title: String,
    contest_id: String,
    summary: String,
    score: f32,
}

// ============================================================================
// Error Handling
// ============================================================================

#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("Gemini API error: {0}")]
    GeminiError(String),

    #[error("Qdrant error: {0}")]
    QdrantError(#[from] qdrant_client::QdrantError),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Lockdown: Practice mode is disabled during contest ({0})")]
    Lockdown(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match &self {
            AppError::GeminiError(msg) => (StatusCode::BAD_GATEWAY, msg.clone()),
            AppError::QdrantError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            AppError::Lockdown(contest_id) => (
                StatusCode::FORBIDDEN,
                format!(
                    "Practice mode is disabled during contest. Contest '{}' is currently running. Please use contest mode instead.",
                    contest_id
                ),
            ),
        };

        if matches!(&self, AppError::Lockdown(_)) {
            warn!("Lockdown access attempt: {}", message);
        } else {
            error!("Request error: {}", message);
        }

        (status, Json(json!({ "error": message }))).into_response()
    }
}

// ============================================================================
// Contest Schedule Fetching
// ============================================================================

async fn fetch_contest_schedule(client: &reqwest::Client) -> Result<Vec<AtCoderContest>, String> {
    let response = client
        .get(CONTESTS_API)
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch contests: {}", e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Contest API returned status: {}",
            response.status()
        ));
    }

    let contests: Vec<AtCoderContest> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse contests: {}", e))?;

    Ok(contests)
}

async fn update_contest_schedule(
    client: &reqwest::Client,
    schedule: &RwLock<ContestSchedule>,
) {
    match fetch_contest_schedule(client).await {
        Ok(contests) => {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let mut schedule = schedule.write().await;
            schedule.contests = contests;
            schedule.last_updated = now;

            info!(
                "Contest schedule updated: {} contests loaded",
                schedule.contests.len()
            );

            // Log if any contest is currently running
            if let Some(contest) = schedule.get_running_contest() {
                warn!(
                    contest_id = %contest.id,
                    "Contest is currently running - Practice mode is LOCKED"
                );
            }
        }
        Err(e) => {
            warn!("Failed to update contest schedule: {}", e);
        }
    }
}

fn spawn_schedule_poller(client: reqwest::Client, schedule: Arc<RwLock<ContestSchedule>>) {
    tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(SCHEDULE_POLL_INTERVAL_SECS));

        loop {
            interval.tick().await;
            update_contest_schedule(&client, &schedule).await;
        }
    });
}

// ============================================================================
// Gemini API
// ============================================================================

#[derive(Debug, Serialize)]
struct GeminiEmbedRequest {
    model: String,
    content: GeminiContent,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbedResponse {
    embedding: GeminiEmbedding,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbedding {
    values: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct GeminiGenerateRequest {
    contents: Vec<GeminiGenerateContent>,
}

#[derive(Debug, Serialize)]
struct GeminiGenerateContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
struct GeminiGenerateResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiCandidateContent,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidateContent {
    parts: Vec<GeminiResponsePart>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponsePart {
    text: String,
}

async fn generate_embedding(
    client: &reqwest::Client,
    api_key: &str,
    text: &str,
) -> Result<Vec<f32>, AppError> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:embedContent?key={}",
        EMBEDDING_MODEL, api_key
    );

    let request = GeminiEmbedRequest {
        model: format!("models/{}", EMBEDDING_MODEL),
        content: GeminiContent {
            parts: vec![GeminiPart {
                text: text.to_string(),
            }],
        },
    };

    let response = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .map_err(|e| AppError::GeminiError(e.to_string()))?;

    if !response.status().is_success() {
        let error_text = response.text().await.unwrap_or_default();
        return Err(AppError::GeminiError(format!(
            "Embedding API failed: {}",
            error_text
        )));
    }

    let embed_response: GeminiEmbedResponse = response
        .json()
        .await
        .map_err(|e| AppError::GeminiError(e.to_string()))?;

    Ok(embed_response.embedding.values)
}

async fn generate_summary(
    client: &reqwest::Client,
    api_key: &str,
    query: &str,
) -> Result<String, AppError> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        GENERATION_MODEL, api_key
    );

    let prompt = format!(
        r#"以下のクエリを、競技プログラミングの問題検索に適した数理的・アルゴリズム的なキーワードに変換してください。
データ構造、アルゴリズム、計算量などの観点から、50-100字程度で簡潔に記述してください。

クエリ: {}

変換結果:"#,
        query
    );

    let request = GeminiGenerateRequest {
        contents: vec![GeminiGenerateContent {
            parts: vec![GeminiPart { text: prompt }],
        }],
    };

    let response = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .map_err(|e| AppError::GeminiError(e.to_string()))?;

    if !response.status().is_success() {
        let error_text = response.text().await.unwrap_or_default();
        return Err(AppError::GeminiError(format!(
            "Generate API failed: {}",
            error_text
        )));
    }

    let gen_response: GeminiGenerateResponse = response
        .json()
        .await
        .map_err(|e| AppError::GeminiError(e.to_string()))?;

    let text = gen_response
        .candidates
        .first()
        .and_then(|c| c.content.parts.first())
        .map(|p| p.text.clone())
        .ok_or_else(|| AppError::GeminiError("No response from Gemini".to_string()))?;

    Ok(text.trim().to_string())
}

// ============================================================================
// Handlers
// ============================================================================

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let schedule = state.contest_schedule.read().await;
    let is_locked = schedule.is_contest_running();
    let running_contest = schedule.get_running_contest().map(|c| c.id.clone());

    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "lockdown": is_locked,
            "running_contest": running_contest
        })),
    )
}

async fn search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, AppError> {
    info!(query = %req.query, mode = ?req.mode, "Search request received");

    // Check lockdown for Practice mode
    if req.mode == SearchMode::Practice {
        let schedule = state.contest_schedule.read().await;
        if let Some(contest) = schedule.get_running_contest() {
            return Err(AppError::Lockdown(contest.id.clone()));
        }
    }

    let results = match req.mode {
        SearchMode::Practice => search_practice(&state, &req.query).await?,
        SearchMode::Contest => search_contest(&state, &req.query).await?,
    };

    Ok(Json(SearchResponse { results }))
}

async fn search_practice(state: &AppState, query: &str) -> Result<Vec<SearchResult>, AppError> {
    // Step 1: クエリを数理的要約に変換
    info!("Generating summary for query...");
    let summary = generate_summary(&state.http_client, &state.gemini_api_key, query).await?;
    info!(summary = %summary, "Summary generated");

    // Step 2: 要約をベクトル化
    info!("Generating embedding...");
    let embedding = generate_embedding(&state.http_client, &state.gemini_api_key, &summary).await?;

    if embedding.len() != EMBEDDING_DIM {
        return Err(AppError::Internal(format!(
            "Unexpected embedding dimension: {} (expected {})",
            embedding.len(),
            EMBEDDING_DIM
        )));
    }

    // Step 3: Qdrantでベクトル検索
    info!("Searching Qdrant...");
    let search_result = state
        .qdrant
        .search_points(SearchPoints {
            collection_name: COLLECTION_NAME.to_string(),
            vector: embedding,
            limit: 10,
            with_payload: Some(WithPayloadSelector {
                selector_options: Some(SelectorOptions::Include(PayloadIncludeSelector {
                    fields: vec![
                        "problem_id".to_string(),
                        "title".to_string(),
                        "contest_id".to_string(),
                        "summary".to_string(),
                    ],
                })),
            }),
            ..Default::default()
        })
        .await?;

    let results = search_result
        .result
        .into_iter()
        .map(|point| {
            let payload = point.payload;
            SearchResult {
                problem_id: payload
                    .get("problem_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                title: payload
                    .get("title")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                contest_id: payload
                    .get("contest_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                summary: payload
                    .get("summary")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                score: point.score,
            }
        })
        .collect();

    Ok(results)
}

async fn search_contest(state: &AppState, query: &str) -> Result<Vec<SearchResult>, AppError> {
    // Contest Mode: summaryフィールドに対するキーワード検索
    // Qdrantのテキストマッチフィルターを使用
    info!("Searching Qdrant with keyword filter...");

    // キーワードを空白で分割して各キーワードでフィルタ
    let keywords: Vec<&str> = query.split_whitespace().collect();

    if keywords.is_empty() {
        return Ok(vec![]);
    }

    // 各キーワードに対してtext match条件を作成（OR検索）
    // フルテキストインデックスを使用
    let conditions: Vec<Condition> = keywords
        .iter()
        .map(|kw| Condition::matches_text("summary", kw.to_string()))
        .collect();

    // ダミーベクトルでスクロール検索（キーワードフィルタのみ）
    // Qdrantはベクトルなしの純粋なフィルタ検索もサポートしているが、
    // SearchPointsを使う場合はダミーベクトルが必要
    let dummy_vector = vec![0.0f32; EMBEDDING_DIM];

    let search_result = state
        .qdrant
        .search_points(SearchPoints {
            collection_name: COLLECTION_NAME.to_string(),
            vector: dummy_vector,
            limit: 10,
            filter: Some(Filter {
                should: conditions, // OR条件
                ..Default::default()
            }),
            with_payload: Some(WithPayloadSelector {
                selector_options: Some(SelectorOptions::Include(PayloadIncludeSelector {
                    fields: vec![
                        "problem_id".to_string(),
                        "title".to_string(),
                        "contest_id".to_string(),
                        "summary".to_string(),
                    ],
                })),
            }),
            ..Default::default()
        })
        .await?;

    let results = search_result
        .result
        .into_iter()
        .map(|point| {
            let payload = point.payload;
            SearchResult {
                problem_id: payload
                    .get("problem_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                title: payload
                    .get("title")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                contest_id: payload
                    .get("contest_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                summary: payload
                    .get("summary")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                score: point.score,
            }
        })
        .collect();

    Ok(results)
}

// ============================================================================
// Router
// ============================================================================

fn build_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health))
        .route("/search", post(search))
        .layer(cors)
        .with_state(state)
}

fn init_logging() {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(fmt::layer().json())
        .init();
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() {
    init_logging();

    // Load .env file (optional, for local development)
    let _ = dotenvy::dotenv();

    // Load environment variables
    let gemini_api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let qdrant_url = std::env::var("QDRANT_URL").expect("QDRANT_URL must be set");
    let qdrant_api_key = std::env::var("QDRANT_API_KEY").expect("QDRANT_API_KEY must be set");

    // Initialize Qdrant client
    info!("Connecting to Qdrant...");
    let qdrant = Qdrant::from_url(&qdrant_url)
        .api_key(qdrant_api_key)
        .build()
        .expect("Failed to create Qdrant client");

    // Verify connection
    match qdrant.health_check().await {
        Ok(_) => info!("Qdrant connection established"),
        Err(e) => {
            warn!("Qdrant health check failed: {}", e);
        }
    }

    // Initialize contest schedule
    let http_client = reqwest::Client::new();
    let contest_schedule = Arc::new(RwLock::new(ContestSchedule::default()));

    // Fetch initial schedule
    info!("Fetching initial contest schedule...");
    update_contest_schedule(&http_client, &contest_schedule).await;

    // Start background schedule poller
    spawn_schedule_poller(http_client.clone(), contest_schedule.clone());

    let state = AppState {
        qdrant: Arc::new(qdrant),
        gemini_api_key,
        http_client,
        contest_schedule,
    };

    let app = build_router(state);
    let addr = "0.0.0.0:8080";

    info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

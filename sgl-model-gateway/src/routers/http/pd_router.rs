use std::{
    borrow::Cow,
    fmt,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use memchr::memmem;
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::time::Instant as TokioInstant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::{
    p2p_node_gate::{P2pAdmissionResult, P2pFreshPlan, P2pNodeGate, P2pNodeLease},
    pd_types::api_path,
};
use crate::{
    config::{types::RetryConfig, PolicyConfig},
    core::{
        is_retryable_status, HashRing, RetryExecutor, Worker, WorkerLoadGuard, WorkerRegistry,
        WorkerType, UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    policies::{
        LoadBalancingPolicy, P2pCacheAwareSelector, P2pPreparedRequest, P2pRoutingConfig,
        PolicyRegistry, RemoteKvDecision, SelectWorkerInfo,
    },
    protocols::{
        chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        classify::ClassifyRequest,
        common::{InputIds, StringOrArray},
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
    },
    routers::{
        error,
        grpc::utils::{
            error_type_from_status, filter_chat_request_by_tool_choice, process_chat_messages,
            route_to_endpoint,
        },
        header_utils,
        streaming_utils::BreakerTrackedStream,
        RouterTrait,
    },
    tokenizer::TokenizerRegistry,
};

pub struct PDRouter {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub client: Client,
    pub retry_config: RetryConfig,
    pub api_key: Option<String>,
    pub enable_igw: bool,
    tokenizer_registry: Arc<TokenizerRegistry>,
    p2p_untruncated_tokenizer: Option<P2pUntruncatedTokenizer>,
    p2p_selector: Option<Arc<P2pCacheAwareSelector>>,
    p2p_node_gate: Option<P2pNodeGate>,
}

impl fmt::Debug for PDRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PDRouter")
            .field("enable_igw", &self.enable_igw)
            .field("p2p_enabled", &self.p2p_selector.is_some())
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
struct PDRequestContext<'a> {
    route: &'static str,
    batch_size: Option<usize>,
    is_stream: bool,
    return_logprob: bool,
    request_text: Option<String>,
    request_tokens: Option<Vec<u32>>,
    model_id: Option<&'a str>,
    headers: Option<HeaderMap>,
}

#[derive(Clone)]
struct P2pUntruncatedTokenizer {
    source: String,
    tokenizer: Arc<tokenizers::Tokenizer>,
}

#[derive(Clone, Serialize)]
struct ForwardedChatRequest<'a> {
    #[serde(flatten)]
    request: &'a ChatCompletionRequest,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_ids: Option<&'a [u32]>,
}

struct SelectedPdPair {
    prefill: Arc<dyn Worker>,
    decode: Arc<dyn Worker>,
    remote_kv: Option<RemoteKvDecision>,
    prepared_p2p_request: Option<P2pPreparedRequest>,
}

const P2P_NODE_LOCK_REPLAN_INTERVAL: Duration = Duration::from_secs(1);
const P2P_NODE_LOCK_ADMISSION_BUDGET: Duration = Duration::from_secs(120);
const P2P_NODE_LOCK_MAX_REPLANS: usize = 120;

struct P2pTransferLease {
    _node_lease: P2pNodeLease,
    explicitly_settled: bool,
    acquired_at: Instant,
    source_url: String,
    target_url: String,
    matched_tokens: usize,
    attempt_id: String,
}

#[derive(Debug, PartialEq, Eq)]
enum P2pTransferOutcome {
    Transferred { transferred_tokens: usize },
    Fallback,
    TransportUncertain,
}

impl P2pTransferLease {
    fn release(mut self, reason: &'static str) {
        self.explicitly_settled = true;
        info!(
            attempt_id = %self.attempt_id,
            source = %self.source_url,
            target = %self.target_url,
            matched_tokens = self.matched_tokens,
            held_ms = self.acquired_at.elapsed().as_millis() as u64,
            reason,
            "P2P source and target node locks released"
        );
    }
}

impl Drop for P2pTransferLease {
    fn drop(&mut self) {
        if !self.explicitly_settled {
            error!(
                attempt_id = %self.attempt_id,
                source = %self.source_url,
                target = %self.target_url,
                matched_tokens = self.matched_tokens,
                "P2P transfer lease dropped without explicit settlement; its two node locks are being released defensively"
            );
        }
    }
}

/// Marker placed on a `Response` by paths inside
/// `execute_dual_dispatch_internal` that have already recorded prefill and
/// decode breaker outcomes against the workers' actual per-side results
/// (rather than the final response status). The outer dispatcher reads this
/// and skips its own status-based `record_outcome` calls so a decode-only
/// transport failure can't be misattributed to a healthy prefill.
#[derive(Clone, Copy)]
struct BreakerOutcomesRecorded;

impl PDRouter {
    fn load_p2p_untruncated_tokenizer(
        source: &str,
    ) -> Result<(tokenizers::Tokenizer, Option<usize>), String> {
        let source_path = Path::new(source);
        let tokenizer_path = if source_path.is_dir() {
            source_path.join("tokenizer.json")
        } else {
            source_path.to_path_buf()
        };
        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            format!(
                "failed to load P2P tokenizer {}: {error}",
                tokenizer_path.display()
            )
        })?;
        let previous_max_length = tokenizer.get_truncation().map(|params| params.max_length);
        tokenizer
            .with_truncation(None)
            .map_err(|error| format!("failed to disable P2P tokenizer truncation: {error}"))?;
        Ok((tokenizer, previous_max_length))
    }

    async fn proxy_to_first_prefill_worker(
        &self,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let workers = self.worker_registry.get_prefill_workers();
        let first_worker_url = workers.first().map(|w| w.url().to_string());

        if let Some(worker_url) = first_worker_url {
            self.proxy_to_worker(worker_url, endpoint, headers).await
        } else {
            error::service_unavailable("no_prefill_servers", "No prefill servers available")
        }
    }

    async fn proxy_to_worker(
        &self,
        worker_url: String,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let url = format!("{}/{}", worker_url, endpoint);
        let mut request_builder = self.client.get(&url);

        if let Some(headers) = headers {
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }
        }

        match request_builder.send().await {
            Ok(res) if res.status().is_success() => {
                let response_headers = header_utils::preserve_response_headers(res.headers());

                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(Body::from(body));
                        *response.status_mut() = StatusCode::OK;
                        *response.headers_mut() = response_headers;
                        response
                    }
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        error::internal_error(
                            "read_response_body_failed",
                            format!("Failed to read response body: {}", e),
                        )
                    }
                }
            }
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                // Use the status code to determine which error function to use
                match status {
                    StatusCode::BAD_REQUEST => error::bad_request(
                        "server_bad_request",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::NOT_FOUND => error::not_found(
                        "server_not_found",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                        "server_internal_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                        "server_unavailable",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::BAD_GATEWAY => error::bad_gateway(
                        "server_bad_gateway",
                        format!("Server returned status: {}", res.status()),
                    ),
                    _ => error::internal_error(
                        "server_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                }
            }
            Err(e) => {
                error!("Failed to proxy request server: {}", e);
                error::internal_error(
                    "proxy_request_failed",
                    format!("Failed to proxy request: {}", e),
                )
            }
        }
    }

    pub async fn new(ctx: &Arc<crate::app_context::AppContext>) -> Result<Self, String> {
        let p2p_selector = ctx.kv_event_index.as_ref().and_then(|index| {
            match ctx
                .router_config
                .mode
                .get_prefill_policy(&ctx.router_config.policy)
            {
                PolicyConfig::CacheAware {
                    cache_threshold,
                    balance_abs_threshold,
                    balance_rel_threshold,
                    ..
                } => Some(Arc::new(P2pCacheAwareSelector::new(
                    P2pRoutingConfig {
                        cache_threshold: *cache_threshold,
                        balance_abs_threshold: *balance_abs_threshold,
                        balance_rel_threshold: *balance_rel_threshold,
                    },
                    index.tree(),
                    index.block_size_oracle(),
                ))),
                policy => {
                    warn!(
                        ?policy,
                        "Prefill P2P requires cache_aware Prefill policy; using legacy routing"
                    );
                    None
                }
            }
        });

        let p2p_untruncated_tokenizer = p2p_selector.as_ref().and_then(|_| {
            let source = ctx
                .router_config
                .tokenizer_path
                .as_ref()
                .or(ctx.router_config.model_path.as_ref())?;
            match Self::load_p2p_untruncated_tokenizer(source) {
                Ok((tokenizer, previous_max_length)) => {
                    info!(
                        model = source,
                        ?previous_max_length,
                        "Loaded untruncated tokenizer for P2P prefix routing"
                    );
                    Some(P2pUntruncatedTokenizer {
                        source: source.clone(),
                        tokenizer: Arc::new(tokenizer),
                    })
                }
                Err(error) => {
                    warn!(
                        model = source,
                        %error,
                        "Unable to load untruncated P2P tokenizer; using registered tokenizer fallback"
                    );
                    None
                }
            }
        });
        let p2p_node_gate = p2p_selector
            .as_ref()
            .map(|_| P2pNodeGate::new(P2P_NODE_LOCK_REPLAN_INTERVAL, P2P_NODE_LOCK_MAX_REPLANS));

        Ok(PDRouter {
            worker_registry: Arc::clone(&ctx.worker_registry),
            policy_registry: Arc::clone(&ctx.policy_registry),
            client: ctx.client.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            api_key: ctx.router_config.api_key.clone(),
            enable_igw: ctx.router_config.enable_igw,
            tokenizer_registry: Arc::clone(&ctx.tokenizer_registry),
            p2p_untruncated_tokenizer,
            p2p_selector,
            p2p_node_gate,
        })
    }

    fn handle_server_selection_error(error: String) -> Response {
        error!("Failed to select PD pair error={}", error);
        error::service_unavailable(
            "server_selection_failed",
            format!("No available servers: {}", error),
        )
    }

    fn handle_serialization_error(error: impl fmt::Display) -> Response {
        error!("Failed to serialize request error={}", error);
        error::internal_error("serialization_failed", "Failed to serialize request")
    }

    fn get_generate_batch_size(req: &GenerateRequest) -> Option<usize> {
        // GenerateRequest doesn't support batch via arrays, only via input_ids
        if let Some(InputIds::Batch(batches)) = &req.input_ids {
            if !batches.is_empty() {
                return Some(batches.len());
            }
        }
        None
    }

    fn get_chat_batch_size(req: &ChatCompletionRequest) -> Option<usize> {
        if let Some(n) = req.n {
            if n > 1 {
                return Some(n as usize);
            }
        }
        None
    }

    fn get_completion_batch_size(req: &CompletionRequest) -> Option<usize> {
        if let StringOrArray::Array(arr) = &req.prompt {
            if !arr.is_empty() {
                return Some(arr.len());
            }
        }
        None
    }

    fn p2p_tokens_from_input_ids(input_ids: &InputIds) -> Option<Vec<u32>> {
        let ids = match input_ids {
            InputIds::Single(ids) => ids.as_slice(),
            InputIds::Batch(batches) if batches.len() == 1 => batches[0].as_slice(),
            InputIds::Batch(_) => return None,
        };
        ids.iter()
            .map(|&id| u32::try_from(id))
            .collect::<Result<Vec<_>, _>>()
            .ok()
    }

    fn resolve_p2p_tokenizer_model_id<'a>(
        &self,
        requested_model: Option<&'a str>,
    ) -> Option<Cow<'a, str>> {
        let requested_model = requested_model?;
        if self.tokenizer_registry.get(requested_model).is_some() {
            return Some(Cow::Borrowed(requested_model));
        }
        if self.enable_igw {
            warn!(
                requested_model,
                "p2p_tokenizer_missing: IGW requires an exact tokenizer model match"
            );
            return None;
        }

        let mut resolved_model: Option<String> = None;
        for worker in self.worker_registry.get_prefill_workers() {
            let worker_model = worker.model_id();
            if worker_model == UNKNOWN_MODEL_ID
                || self.tokenizer_registry.get(worker_model).is_none()
            {
                continue;
            }
            match resolved_model.as_deref() {
                None => resolved_model = Some(worker_model.to_string()),
                Some(existing) if existing == worker_model => {}
                Some(existing) => {
                    warn!(
                        requested_model,
                        first_tokenizer_model = existing,
                        conflicting_tokenizer_model = worker_model,
                        "p2p_tokenizer_alias_ambiguous"
                    );
                    return None;
                }
            }
        }
        resolved_model.map(Cow::Owned)
    }

    fn p2p_untruncated_tokenizer_for_model(
        &self,
        model_id: &str,
    ) -> Option<&tokenizers::Tokenizer> {
        let untruncated = self.p2p_untruncated_tokenizer.as_ref()?;
        if !self.enable_igw {
            return Some(untruncated.tokenizer.as_ref());
        }

        let entry = self
            .tokenizer_registry
            .get_by_name(model_id)
            .or_else(|| self.tokenizer_registry.get_by_id(model_id))?;
        (entry.source == untruncated.source).then_some(untruncated.tokenizer.as_ref())
    }

    fn encode_p2p_text(&self, model_id: Option<&str>, text: &str) -> Option<Vec<u32>> {
        let model_id = self.resolve_p2p_tokenizer_model_id(model_id)?;
        if let Some(tokenizer) = self.p2p_untruncated_tokenizer_for_model(model_id.as_ref()) {
            return tokenizer
                .encode(text, false)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|error| {
                    debug!(model = %model_id, %error, "Untruncated P2P tokenization failed");
                    error
                })
                .ok();
        }
        let tokenizer = self.tokenizer_registry.get(model_id.as_ref())?;
        tokenizer
            .encode(text, false)
            .map(|encoding| encoding.token_ids().to_vec())
            .map_err(|err| {
                debug!(model = %model_id, error = %err, "P2P tokenization failed");
                err
            })
            .ok()
    }

    fn p2p_tokens_for_generate(
        &self,
        request: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Option<Vec<u32>> {
        if let Some(input_ids) = request.input_ids.as_ref() {
            return Self::p2p_tokens_from_input_ids(input_ids);
        }
        self.encode_p2p_text(model_id, request.text.as_deref()?)
    }

    fn p2p_tokens_for_chat(
        &self,
        request: &ChatCompletionRequest,
        model_id: Option<&str>,
        input_ids: Option<&[u32]>,
    ) -> Option<Vec<u32>> {
        if let Some(input_ids) = input_ids {
            info!(
                token_source = "upstream",
                token_count = input_ids.len(),
                "P2P routing token source selected"
            );
            return Some(input_ids.to_vec());
        }

        let requested_model = model_id.unwrap_or(&request.model);
        let model_id = self.resolve_p2p_tokenizer_model_id(Some(requested_model))?;
        let tokenizer = self.tokenizer_registry.get(model_id.as_ref())?;
        let filtered_request = filter_chat_request_by_tool_choice(request);
        let processed = process_chat_messages(filtered_request.as_ref(), tokenizer.as_ref())
            .map_err(|err| {
                debug!(model = %model_id, error = %err, "P2P chat-template processing failed");
                err
            })
            .ok()?;
        if let Some(untruncated) = self.p2p_untruncated_tokenizer_for_model(model_id.as_ref()) {
            return untruncated
                .encode(processed.text.as_str(), false)
                .map(|encoding| {
                    let tokens = encoding.get_ids().to_vec();
                    info!(
                        token_source = "local_untruncated",
                        token_count = tokens.len(),
                        "P2P routing token source selected"
                    );
                    tokens
                })
                .map_err(|error| {
                    debug!(model = %model_id, %error, "Untruncated P2P chat tokenization failed");
                    error
                })
                .ok();
        }
        tokenizer
            .encode(&processed.text, false)
            .map(|encoding| {
                let tokens = encoding.token_ids().to_vec();
                info!(
                    token_source = "registered_fallback",
                    token_count = tokens.len(),
                    "P2P routing token source selected"
                );
                tokens
            })
            .ok()
    }

    fn p2p_tokens_for_completion(
        &self,
        request: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Option<Vec<u32>> {
        let prompt = match &request.prompt {
            StringOrArray::String(text) => text.as_str(),
            StringOrArray::Array(texts) => texts.first()?.as_str(),
        };
        self.encode_p2p_text(Some(model_id.unwrap_or(&request.model)), prompt)
    }

    // Static key strings to avoid per-request allocations
    const BOOTSTRAP_HOST_KEY: &'static str = "bootstrap_host";
    const BOOTSTRAP_PORT_KEY: &'static str = "bootstrap_port";
    const BOOTSTRAP_ROOM_KEY: &'static str = "bootstrap_room";
    const REMOTE_KV_KEYS: [&'static str; 7] = [
        "remote_kv_source_url",
        "remote_kv_source_bootstrap_addr",
        "remote_kv_target_url",
        "remote_kv_matched_tokens",
        "remote_kv_token_ids",
        "remote_kv_reason",
        "remote_kv_attempt_id",
    ];
    const REMOTE_KV_HEADER_KEYS: [&'static str; 6] = [
        "x-sgl-remote-kv-source",
        "x-sgl-remote-kv-source-bootstrap-addr",
        "x-sgl-remote-kv-target",
        "x-sgl-remote-kv-matched-tokens",
        "x-sgl-remote-kv-reason",
        "x-sgl-remote-kv-attempt-id",
    ];
    const P2P_CACHE_NAMESPACE_KEYS: [&'static str; 4] =
        ["extra_key", "cache_salt", "lora_id", "lora_path"];

    fn prepare_pd_headers(inbound: &HeaderMap) -> (HeaderMap, HeaderMap) {
        let mut clean = inbound.clone();
        for key in Self::REMOTE_KV_HEADER_KEYS {
            clean.remove(key);
        }
        (clean.clone(), clean)
    }

    fn prepare_pd_payloads(original: &Value) -> Result<(Value, Value), String> {
        let mut clean = original.clone();
        let clean_obj = clean
            .as_object_mut()
            .ok_or_else(|| "Request must be a JSON object".to_string())?;
        for key in Self::REMOTE_KV_KEYS {
            clean_obj.remove(key);
        }

        Ok((clean.clone(), clean))
    }

    fn p2p_transfer_payload(decision: &RemoteKvDecision, attempt_id: &str) -> Value {
        json!({
            "source_url": decision.source_url,
            "target_url": decision.target_url,
            "token_ids": decision.token_ids,
            "matched_tokens": decision.matched_tokens,
            "request_id": attempt_id,
            "dry_run": false,
            "reason": decision.reason,
            "source_bootstrap_addr": decision.source_bootstrap_addr,
        })
    }

    fn p2p_nonempty_cache_namespace(request: &Value) -> Option<&'static str> {
        let request = request.as_object()?;
        Self::P2P_CACHE_NAMESPACE_KEYS
            .into_iter()
            .find(|key| match request.get(*key) {
                None | Some(Value::Null) => false,
                Some(Value::String(value)) => !value.is_empty(),
                Some(Value::Array(value)) => !value.is_empty(),
                Some(Value::Object(value)) => !value.is_empty(),
                Some(_) => true,
            })
    }

    fn p2p_response_endpoint_matches(payload: &Value, field: &str, expected: &str) -> bool {
        match payload.get(field).and_then(Value::as_str) {
            None | Some("") => true,
            Some(actual) => actual.trim_end_matches('/') == expected.trim_end_matches('/'),
        }
    }

    async fn execute_independent_p2p_transfer(
        &self,
        headers: &HeaderMap,
        decision: &RemoteKvDecision,
        lease: P2pTransferLease,
    ) -> P2pTransferOutcome {
        let payload = Self::p2p_transfer_payload(decision, &lease.attempt_id);
        let target_endpoint = decision.target_url.trim_end_matches('/');
        let request = self.build_post_with_headers(
            &self.client,
            target_endpoint,
            "/experimental/p2p_kv_transfer",
            &payload,
            Some(headers),
            false,
        );
        let decision = decision.clone();
        let attempt_id = lease.attempt_id.clone();

        // The worker shields an admitted P2P operation until its data path has
        // settled. Mirror that ownership here: dropping the downstream HTTP
        // handler must not cancel the control request and release both node
        // locks while the worker can still be ACTIVE. Tokio detaches a spawned
        // task when its JoinHandle is dropped, so the task continues to own the
        // request and lease through a client disconnect.
        match tokio::spawn(async move {
            Self::run_independent_p2p_transfer(request, decision, lease).await
        })
        .await
        {
            Ok(outcome) => outcome,
            Err(error) => {
                error!(
                    attempt_id,
                    %error,
                    "Independent P2P control settlement task failed; its lease was released defensively"
                );
                P2pTransferOutcome::TransportUncertain
            }
        }
    }

    async fn run_independent_p2p_transfer(
        request: reqwest::RequestBuilder,
        decision: RemoteKvDecision,
        lease: P2pTransferLease,
    ) -> P2pTransferOutcome {
        info!(
            attempt_id = %lease.attempt_id,
            source = %decision.source_url,
            target = %decision.target_url,
            matched_tokens = decision.matched_tokens,
            "Independent P2P transfer control request started"
        );

        let response = request.send().await;

        let response = match response {
            Ok(response) => response,
            Err(error) => {
                warn!(
                    attempt_id = %lease.attempt_id,
                    source = %decision.source_url,
                    target = %decision.target_url,
                    matched_tokens = decision.matched_tokens,
                    %error,
                    "Independent P2P transport ended without a response; falling back to local recompute and relying on Worker single-flight/quarantine admission for any still-running transfer"
                );
                lease.release("p2p_control_transport_uncertain");
                return P2pTransferOutcome::TransportUncertain;
            }
        };

        let status = response.status();
        let payload = match response.json::<Value>().await {
            Ok(payload) => payload,
            Err(error) => {
                warn!(
                    attempt_id = %lease.attempt_id,
                    source = %decision.source_url,
                    target = %decision.target_url,
                    matched_tokens = decision.matched_tokens,
                    http_status = %status,
                    %error,
                    "Independent P2P control response was not valid JSON; falling back to local recompute"
                );
                lease.release("p2p_control_terminal_invalid_response");
                return P2pTransferOutcome::Fallback;
            }
        };

        let success = payload.get("success").and_then(Value::as_bool) == Some(true);
        let fallback_recompute = payload
            .get("fallback_recompute")
            .and_then(Value::as_bool)
            .unwrap_or(!success);
        let transferred_tokens = payload
            .get("transferred_tokens")
            .and_then(Value::as_u64)
            .and_then(|tokens| usize::try_from(tokens).ok())
            .unwrap_or_default();
        let response_source_matches =
            Self::p2p_response_endpoint_matches(&payload, "source_url", &decision.source_url);
        let response_target_matches =
            Self::p2p_response_endpoint_matches(&payload, "target_url", &decision.target_url);
        let transferred_tokens_valid = transferred_tokens > 0
            && transferred_tokens <= decision.matched_tokens
            && transferred_tokens <= decision.token_ids.len();
        let message = payload
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("worker returned no message");

        if status.is_success()
            && success
            && !fallback_recompute
            && transferred_tokens_valid
            && response_source_matches
            && response_target_matches
        {
            info!(
                attempt_id = %lease.attempt_id,
                source = %decision.source_url,
                target = %decision.target_url,
                matched_tokens = decision.matched_tokens,
                transferred_tokens,
                "Independent P2P transfer control request completed successfully"
            );
            lease.release("p2p_control_terminal_success");
            return P2pTransferOutcome::Transferred { transferred_tokens };
        }

        warn!(
            attempt_id = %lease.attempt_id,
            source = %decision.source_url,
            target = %decision.target_url,
            matched_tokens = decision.matched_tokens,
            transferred_tokens,
            http_status = %status,
            success,
            fallback_recompute,
            transferred_tokens_valid,
            response_source_matches,
            response_target_matches,
            message,
            "Independent P2P transfer reached a terminal fallback; ordinary Prefill will recompute locally"
        );
        lease.release("p2p_control_terminal_fallback");
        P2pTransferOutcome::Fallback
    }

    fn inject_bootstrap_into_value(
        mut original: Value,
        prefill_worker: &dyn Worker,
        batch_size: Option<usize>,
    ) -> Result<Value, String> {
        let obj = original
            .as_object_mut()
            .ok_or_else(|| "Request must be a JSON object".to_string())?;

        if let Some(n) = batch_size {
            let mut hosts = Vec::with_capacity(n);
            let mut ports = Vec::with_capacity(n);
            let mut rooms = Vec::with_capacity(n);
            for _ in 0..n {
                hosts.push(prefill_worker.bootstrap_host());
                ports.push(prefill_worker.bootstrap_port());
                rooms.push(super::pd_types::generate_room_id());
            }
            // Use static string keys to avoid per-request allocations
            obj.insert(
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                Value::Array(hosts.into_iter().map(Value::from).collect()),
            );
            obj.insert(
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                Value::Array(
                    ports
                        .into_iter()
                        .map(|p| match p {
                            Some(v) => Value::from(v),
                            None => Value::Null,
                        })
                        .collect(),
                ),
            );
            obj.insert(
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                Value::Array(rooms.into_iter().map(Value::from).collect()),
            );
        } else {
            // Use static string keys to avoid per-request allocations
            obj.insert(
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                Value::from(prefill_worker.bootstrap_host()),
            );
            obj.insert(
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                match prefill_worker.bootstrap_port() {
                    Some(v) => Value::from(v),
                    None => Value::Null,
                },
            );
            obj.insert(
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                Value::from(super::pd_types::generate_room_id()),
            );
        }
        Ok(original)
    }

    async fn execute_dual_dispatch<T: Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        original_request: &T,
        context: PDRequestContext<'_>,
    ) -> Response {
        let start_time = Instant::now();

        let route = context.route;
        let model = context.model_id.unwrap_or(UNKNOWN_MODEL_ID);
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_PD,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(context.is_stream),
        );
        // Clone request once outside the retry loop, then use Arc to share across attempts
        // This avoids O(retries) clones by sharing the same data
        let shared_request = Arc::new(original_request.clone());
        // Candidate selection, queueing, and stale-plan rejection do not count
        // as a data-plane attempt. Once a real Worker control request is
        // dispatched, later HTTP retries must use local recompute.
        let p2p_dispatched = Arc::new(AtomicBool::new(false));
        // This is one absolute budget for the entire logical request. Lock
        // releases, replan ticks, and outer HTTP retries never extend it.
        let p2p_admission_deadline = TokioInstant::now() + P2P_NODE_LOCK_ADMISSION_BUDGET;
        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            {
                move |attempt: u32| {
                    // Clone Arc (cheap reference count increment) instead of cloning the entire request
                    let shared_request = Arc::clone(&shared_request);
                    let p2p_dispatched = Arc::clone(&p2p_dispatched);
                    let context = context.clone();
                    async move {
                        let selected = match self
                            .select_pd_pair_with_decision(
                                context.request_text.as_deref(),
                                context.request_tokens.as_deref(),
                                context.model_id,
                                context.headers.as_ref(),
                            )
                            .await
                        {
                            Ok(pair) => pair,
                            Err(e) => {
                                return Self::handle_server_selection_error(e);
                            }
                        };

                        let mut prefill = selected.prefill;
                        let mut decode = selected.decode;
                        let mut p2p_decision = selected.remote_kv;
                        let prepared_p2p_request = selected.prepared_p2p_request;
                        if p2p_decision.is_some()
                            && p2p_dispatched.load(Ordering::Acquire)
                        {
                            info!(
                                attempt,
                                "P2P already dispatched for this logical request; retrying with local recompute"
                            );
                            p2p_decision = None;
                        }
                        let mut admitted_node_lease = None;

                        let base_json_request = match serde_json::to_value(shared_request.as_ref()) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        if p2p_decision.is_some() {
                            if let Some(namespace_field) =
                                Self::p2p_nonempty_cache_namespace(&base_json_request)
                            {
                                warn!(
                                    namespace_field,
                                    "P2P transfer skipped because the request uses a cache namespace or LoRA that the Router control payload cannot reproduce safely"
                                );
                                p2p_decision = None;
                            }
                        }

                        if p2p_decision.is_some() {
                            let request_tokens = match context.request_tokens.as_deref() {
                                Some(tokens) => tokens,
                                None => {
                                    error!(
                                        "P2P routing produced a decision without request tokens; \
                                         forcing the fresh planner to stop and use local recompute"
                                    );
                                    &[]
                                }
                            };
                            let p2p_lock_wait_started = Instant::now();
                            info!(
                                replan_interval_ms =
                                    P2P_NODE_LOCK_REPLAN_INTERVAL.as_millis() as u64,
                                admission_budget_ms =
                                    P2P_NODE_LOCK_ADMISSION_BUDGET.as_millis() as u64,
                                "Registered fair P2P waiter; replanning source and target until the absolute admission deadline"
                            );

                            if let Some(gate) = self.p2p_node_gate.as_ref() {
                                loop {
                                    match gate
                                        .acquire_best_with(
                                            p2p_admission_deadline,
                                            || {
                                                self.fresh_p2p_plan(
                                                    request_tokens,
                                                    prepared_p2p_request.as_ref(),
                                                    context.model_id,
                                                )
                                            },
                                            |expected_source, target| {
                                                self.validate_granted_p2p_target(
                                                    request_tokens,
                                                    prepared_p2p_request.as_ref(),
                                                    context.model_id,
                                                    expected_source,
                                                    target,
                                                )
                                            },
                                        )
                                        .await
                                    {
                                        P2pAdmissionResult::Granted {
                                            context: fresh_decision,
                                            target,
                                            lease,
                                        } => {
                                            info!(
                                                source = fresh_decision.source_url,
                                                target = fresh_decision.target_url,
                                                matched_tokens = fresh_decision.matched_tokens,
                                                wait_ms = p2p_lock_wait_started
                                                    .elapsed()
                                                    .as_millis()
                                                    as u64,
                                                "Fair P2P admission atomically selected and locked the current lowest-load unlocked target"
                                            );
                                            prefill = target;
                                            p2p_decision = Some(fresh_decision);
                                            admitted_node_lease = Some(lease);

                                            // Decode selection is refreshed after
                                            // admission so a long queue wait does
                                            // not freeze both halves of the route.
                                            let refreshed = match self
                                                .select_pd_pair_with_decision(
                                                    context.request_text.as_deref(),
                                                    None,
                                                    context.model_id,
                                                    context.headers.as_ref(),
                                                )
                                                .await
                                            {
                                                Ok(pair) => pair,
                                                Err(error) => {
                                                    return Self::handle_server_selection_error(
                                                        error,
                                                    );
                                                }
                                            };
                                            decode = refreshed.decode;
                                            break;
                                        }
                                        P2pAdmissionResult::Stopped { reason, fallback } => {
                                            info!(
                                                reason,
                                                wait_ms = p2p_lock_wait_started
                                                    .elapsed()
                                                    .as_millis()
                                                    as u64,
                                                "Fresh Router state no longer admits P2P; using a freshly selected local Prefill route"
                                            );
                                            let refreshed = match self
                                                .select_pd_pair_with_decision(
                                                    context.request_text.as_deref(),
                                                    None,
                                                    context.model_id,
                                                    context.headers.as_ref(),
                                                )
                                                .await
                                            {
                                                Ok(pair) => pair,
                                                Err(error) => {
                                                    return Self::handle_server_selection_error(
                                                        error,
                                                    );
                                                }
                                            };
                                            prefill = fallback
                                                .filter(|worker| worker.is_available())
                                                .unwrap_or(refreshed.prefill);
                                            decode = refreshed.decode;
                                            p2p_decision = None;
                                            break;
                                        }
                                        P2pAdmissionResult::TimedOut => {
                                            warn!(
                                                wait_ms = p2p_lock_wait_started
                                                    .elapsed()
                                                    .as_millis()
                                                    as u64,
                                                admission_budget_ms =
                                                    P2P_NODE_LOCK_ADMISSION_BUDGET
                                                        .as_millis()
                                                        as u64,
                                                "P2P fair admission budget exhausted; using a freshly selected local Prefill route"
                                            );
                                            let refreshed = match self
                                                .select_pd_pair_with_decision(
                                                    context.request_text.as_deref(),
                                                    None,
                                                    context.model_id,
                                                    context.headers.as_ref(),
                                                )
                                                .await
                                            {
                                                Ok(pair) => pair,
                                                Err(error) => {
                                                    return Self::handle_server_selection_error(
                                                        error,
                                                    );
                                                }
                                            };
                                            prefill = refreshed.prefill;
                                            decode = refreshed.decode;
                                            p2p_decision = None;
                                            break;
                                        }
                                    }
                                }
                            } else {
                                error!(
                                    "P2P routing produced a transfer without a node-lock gate; \
                                     falling back to local recompute"
                                );
                                p2p_decision = None;
                            }
                        }

                        debug!(
                            "PD retry attempt {} using final prefill={} decode={}",
                            attempt,
                            prefill.url(),
                            decode.url()
                        );

                        // Bootstrap metadata and both normal payloads are built
                        // only after admission settles, so a replanned Prefill
                        // can never inherit the old target's bootstrap room.
                        let json_request = match Self::inject_bootstrap_into_value(
                            base_json_request,
                            prefill.as_ref(),
                            context.batch_size,
                        ) {
                            Ok(value) => value,
                            Err(error) => {
                                return Self::handle_serialization_error(error);
                            }
                        };
                        let (prefill_json_request, decode_json_request) =
                            match Self::prepare_pd_payloads(&json_request) {
                                Ok(values) => values,
                                Err(error) => {
                                    return Self::handle_serialization_error(error);
                                }
                            };

                        // Worker KV reservation begins only after this control
                        // dispatch. Once dispatched, the lease remains owned by
                        // the shielded settlement task through terminal state.
                        if let (Some(decision), Some(node_lease)) =
                            (p2p_decision.as_ref(), admitted_node_lease.take())
                        {
                            if p2p_dispatched
                                .compare_exchange(
                                    false,
                                    true,
                                    Ordering::AcqRel,
                                    Ordering::Acquire,
                                )
                                .is_err()
                            {
                                info!(
                                    attempt,
                                    "Another retry already dispatched P2P; releasing the unused admission lease"
                                );
                                drop(node_lease);
                            } else {
                                let acquired_at = Instant::now();
                                let attempt_id = format!("router-p2p-{}", Uuid::new_v4());
                                let source = decision.source_url.as_str();
                                let target = decision.target_url.as_str();
                                let matched_tokens = decision.matched_tokens;
                                let lease = P2pTransferLease {
                                    _node_lease: node_lease,
                                    explicitly_settled: false,
                                    acquired_at,
                                    source_url: source.to_string(),
                                    target_url: target.to_string(),
                                    matched_tokens,
                                    attempt_id: attempt_id.clone(),
                                };
                                let mut p2p_headers = headers.cloned().unwrap_or_default();
                                for key in Self::REMOTE_KV_HEADER_KEYS {
                                    p2p_headers.remove(key);
                                }
                                inject_trace_context_http(&mut p2p_headers);
                                info!(
                                    attempt_id,
                                    source,
                                    target,
                                    matched_tokens,
                                    "Dispatching the admitted P2P control request"
                                );
                                let _ = self
                                    .execute_independent_p2p_transfer(
                                        &p2p_headers,
                                        decision,
                                        lease,
                                    )
                                    .await;
                            }
                        } else if admitted_node_lease.is_some() {
                            error!(
                                "P2P admission retained a node lease without a final decision; releasing it before local recompute"
                            );
                            drop(admitted_node_lease.take());
                        }

                        let ctx_is_stream = context.is_stream;
                        let response = self
                            .execute_dual_dispatch_internal(
                                headers,
                                prefill_json_request,
                                decode_json_request,
                                context,
                                Arc::clone(&prefill),
                                Arc::clone(&decode),
                                start_time,
                            )
                            .await;

                        let status = response.status();
                        let outcomes_already_recorded = response
                            .extensions()
                            .get::<BreakerOutcomesRecorded>()
                            .is_some();
                        if !outcomes_already_recorded {
                            let not_error = status.is_success() || status.is_client_error();
                            // Prefill is always non-streaming and fully read before
                            // we get here, so its outcome is final.
                            prefill.record_outcome(not_error);
                            // Decode for a streaming request is still mid-flight at
                            // this point; the `BreakerTrackedStream` wrapped around
                            // its byte stream records the outcome on drop. Skip the
                            // eager success record to avoid masking "200-then-broken"
                            // decode workers.
                            if !ctx_is_stream {
                                decode.record_outcome(not_error);
                            }
                        }

                        // Record worker errors for server errors (5xx)
                        if status.is_server_error() {
                            let error_type = error_type_from_status(status);
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_PREFILL,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_DECODE,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                        }

                        response
                    }
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                // Layer 3 worker metrics (PD mode uses both prefill and decode workers)
                Metrics::record_worker_retry(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retry(metrics_labels::WORKER_DECODE, endpoint);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_DECODE, endpoint);
            },
        )
        .await;

        // Record Layer 2 metrics
        let duration = start_time.elapsed();
        if response.status().is_success() {
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_status(response.status()) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    async fn handle_decode_error_response(
        &self,
        res: reqwest::Response,
        context: &PDRequestContext<'_>,
        decode: Arc<dyn Worker>,
        guards: Vec<WorkerLoadGuard>,
    ) -> Response {
        let status = res.status();

        if context.is_stream {
            // Handle streaming error response
            let response_headers = header_utils::preserve_response_headers(res.headers());
            let error_payload = match res.bytes().await {
                Ok(error_body) => match serde_json::from_slice::<Value>(&error_body) {
                    Ok(error_json) => {
                        json!({ "message": error_json, "status": status.as_u16() })
                    }
                    Err(parse_err) => {
                        let body_text = String::from_utf8_lossy(&error_body).to_string();
                        let preview: String = body_text.chars().take(256).collect();
                        tracing::warn!(
                            "Failed to parse decode error body as JSON from {}: {} \
                             (status={}, body preview: {:?})",
                            decode.url(),
                            parse_err,
                            status.as_u16(),
                            preview
                        );
                        json!({ "message": body_text, "status": status.as_u16() })
                    }
                },
                Err(e) => {
                    json!({ "message": format!("Decode server error: {}", e), "status": status.as_u16() })
                }
            };

            let sse_data = format!(
                "data: {{'error': {}}}",
                serde_json::to_string(&error_payload).unwrap_or_default()
            );
            let error_stream = tokio_stream::once(Ok(axum::body::Bytes::from(sse_data)));

            self.create_streaming_response(
                error_stream,
                status,
                None,
                context.return_logprob,
                Some(response_headers),
                decode,
                guards,
            )
        } else {
            // Handle non-streaming error response
            match res.bytes().await {
                Ok(error_body) => {
                    // Try to parse error message from body, fallback to status-based error
                    let error_message = if let Ok(error_json) =
                        serde_json::from_slice::<Value>(&error_body)
                    {
                        if let Some(msg) = error_json
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else if let Some(msg) = error_json.get("message").and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else {
                            String::from_utf8_lossy(&error_body).to_string()
                        }
                    } else {
                        String::from_utf8_lossy(&error_body).to_string()
                    };

                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_bad_request", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_not_found", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_internal_error", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_unavailable", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_bad_gateway", error_message)
                        }
                        _ => error::internal_error("decode_error", error_message),
                    }
                }
                Err(e) => {
                    let error_message = format!("Decode server error: {}", e);
                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_read_failed", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_read_failed", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_read_failed", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_read_failed", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_read_failed", error_message)
                        }
                        _ => error::internal_error("decode_read_failed", error_message),
                    }
                }
            }
        }
    }

    // Internal method that performs the actual dual dispatch (without retry logic)
    #[allow(clippy::too_many_arguments)]
    async fn execute_dual_dispatch_internal(
        &self,
        headers: Option<&HeaderMap>,
        prefill_json_request: Value,
        decode_json_request: Value,
        context: PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        _start_time: Instant,
    ) -> Response {
        // Create load guards before dispatch so PD cache-aware routing can observe
        // in-flight work as soon as the request is sent to prefill/decode workers.
        let guards = vec![
            WorkerLoadGuard::new(prefill.clone(), headers),
            WorkerLoadGuard::new(decode.clone(), headers),
        ];

        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let (prefill_headers, decode_headers) = Self::prepare_pd_headers(&headers_with_trace);

        // Build both requests
        let prefill_request = self.build_post_with_headers(
            &self.client,
            prefill.url(),
            context.route,
            &prefill_json_request,
            Some(&prefill_headers),
            false,
        );
        let decode_request = self.build_post_with_headers(
            &self.client,
            decode.url(),
            context.route,
            &decode_json_request,
            Some(&decode_headers),
            false,
        );

        // Run both in this handler task (not a detached tokio::spawn) so a client
        // disconnect cancels the pending decode request too, keeping the
        // upstream-cancel behavior from #19524.
        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        let prefill_fut = prefill_request.send();
        let decode_fut = decode_request.send();
        tokio::pin!(prefill_fut);
        tokio::pin!(decode_fut);

        // Poll both until prefill resolves; decode normally resolves later, but
        // may resolve first if it rejects the request outright.
        let prefill_result;
        let mut decode_early: Option<Result<reqwest::Response, reqwest::Error>> = None;
        loop {
            tokio::select! {
                biased;
                pr = &mut prefill_fut => {
                    prefill_result = pr;
                    break;
                }
                dr = &mut decode_fut, if decode_early.is_none() => {
                    decode_early = Some(dr);
                }
            }
        }

        // Decode can't generate without prefill's KV, so any prefill failure
        // (non-2xx / transport error) dooms the paired decode request, which would
        // otherwise block in WaitingForInput until the 300s disaggregation
        // timeout. Drop the decode future to close its connection; the decode
        // engine then detects the disconnect and aborts the request in ~4-8s.
        let prefill_failed = match &prefill_result {
            Ok(resp) => !resp.status().is_success(),
            Err(_) => true,
        };

        if prefill_failed {
            warn!(
                "Prefill failed, aborting paired decode request decode_url={} prefill_url={}",
                decode.url(),
                prefill.url()
            );

            // Tick prefill by its real status (4xx = client fault). Don't record
            // decode: it was cancelled due to a prefill fault, not its own, so a
            // prefill error storm can't trip healthy decode breakers.
            let prefill_ok = match &prefill_result {
                Ok(r) => r.status().is_client_error(),
                Err(_) => false,
            };
            prefill.record_outcome(prefill_ok);

            // Status-faithful error shaping (4xx forwarded, transport/5xx -> 502).
            let mut response =
                match Self::process_prefill_response(prefill_result, prefill.url(), false).await {
                    Err(error_response) => error_response,
                    Ok(_) => error::bad_gateway(
                        "prefill_server_error",
                        "Prefill reported failure but returned a success response".to_string(),
                    ),
                };
            response.extensions_mut().insert(BreakerOutcomesRecorded);
            return response;
        }

        // Prefill ok: take decode's result, awaiting it if still pending.
        let decode_result = match decode_early {
            Some(dr) => dr,
            None => (&mut decode_fut).await,
        };

        events::RequestReceivedEvent {}.emit();

        // Process decode response
        match decode_result {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                debug!("Decode response status: {}", status);

                if !status.is_success() {
                    error!(
                        "Decode server returned error status decode_url={} status={}",
                        decode.url(),
                        status
                    );

                    // Per-worker breaker attribution before the synthetic 5xx
                    // response takes over. Prefill ran concurrently in the
                    // `tokio::join!`: tick it based on its actual response
                    // status, not on the decode-driven failure. For
                    // non-streaming the response carries no tracked stream
                    // so record decode's outcome here too — but treat 4xx
                    // as a client fault rather than a worker fault, matching
                    // the legacy outer-dispatcher rule and the streaming
                    // `BreakerTrackedStream` pre-mark in
                    // `create_streaming_response`. For streaming
                    // `handle_decode_error_response` wraps the synthetic
                    // error SSE in a `BreakerTrackedStream` that ticks
                    // decode on drop, so skip to avoid double-counting.
                    // Mark the response so the outer dispatcher skips its
                    // status-derived `record_outcome`.
                    let prefill_ok = match &prefill_result {
                        Ok(r) => {
                            let s = r.status();
                            s.is_success() || s.is_client_error()
                        }
                        Err(_) => false,
                    };
                    prefill.record_outcome(prefill_ok);
                    if !context.is_stream {
                        let decode_ok = status.is_success() || status.is_client_error();
                        decode.record_outcome(decode_ok);
                    }

                    let mut response = self
                        .handle_decode_error_response(res, &context, decode, guards)
                        .await;
                    response.extensions_mut().insert(BreakerOutcomesRecorded);
                    return response;
                }

                // Process prefill response
                let prefill_body = if context.return_logprob {
                    match Self::process_prefill_response(
                        prefill_result,
                        prefill.url(),
                        context.return_logprob,
                    )
                    .await
                    {
                        Ok((_, body)) => body,
                        Err(error_response) => return error_response,
                    }
                } else {
                    // Even if we don't need logprobs, we should check prefill status
                    match Self::process_prefill_response(prefill_result, prefill.url(), false).await
                    {
                        Ok((_, body)) => body,
                        Err(error_response) => return error_response,
                    }
                };

                if context.is_stream {
                    // Streaming response
                    let prefill_logprobs = if context.return_logprob {
                        prefill_body
                            .as_ref()
                            .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                            .and_then(|json| {
                                json.pointer("/meta_info/input_token_logprobs").cloned()
                            })
                    } else {
                        None
                    };

                    let response_headers = header_utils::preserve_response_headers(res.headers());

                    self.create_streaming_response(
                        res.bytes_stream(),
                        status,
                        prefill_logprobs,
                        context.return_logprob,
                        Some(response_headers),
                        decode,
                        guards,
                    )
                } else {
                    // Non-streaming response
                    if context.return_logprob {
                        self.process_non_streaming_response(
                            res,
                            status,
                            context.return_logprob,
                            prefill_body,
                        )
                        .await
                    } else {
                        // Direct passthrough when no logprobs needed
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(decode_body) => {
                                let mut response = Response::new(Body::from(decode_body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => {
                                error!("Failed to read decode response: {}", e);
                                error::internal_error(
                                    "read_response_failed",
                                    "Failed to read response",
                                )
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!(
                    decode_url = %decode.url(),
                    error = %e,
                    "Decode request failed"
                );
                // Decode failed at TCP/transport level. No tracked
                // stream will ever wrap a response (streaming path) and
                // we shortcut past the outer non-streaming
                // `record_outcome` too — so record decode failure
                // directly. Prefill ran concurrently in the
                // `tokio::join!`: record its real per-worker outcome
                // (success on a 2xx/4xx send, failure on transport
                // error) so the decode-driven 502 doesn't penalise a
                // healthy prefill. Mark the response so the outer
                // dispatcher skips its status-derived `record_outcome`
                // and we don't double-count.
                decode.record_outcome(false);
                let prefill_ok = match &prefill_result {
                    Ok(res) => {
                        let s = res.status();
                        s.is_success() || s.is_client_error()
                    }
                    Err(_) => false,
                };
                prefill.record_outcome(prefill_ok);

                let mut response = error::bad_gateway(
                    "decode_server_error",
                    format!("Decode server error: {}", e),
                );
                response.extensions_mut().insert(BreakerOutcomesRecorded);
                response
            }
        }
    }

    fn policies_need_request_text(&self) -> bool {
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        prefill_policy.needs_request_text() || decode_policy.needs_request_text()
    }

    fn available_p2p_prefills(&self, model_id: Option<&str>) -> Vec<Arc<dyn Worker>> {
        let effective_model_id = if !self.enable_igw { None } else { model_id };
        let workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|worker| matches!(worker.worker_type(), WorkerType::Prefill { .. }))
                .cloned()
                .collect::<Vec<_>>()
        } else {
            self.worker_registry.get_prefill_workers()
        };
        workers
            .into_iter()
            .filter(|worker| worker.is_available())
            .collect()
    }

    fn fresh_p2p_plan(
        &self,
        request_tokens: &[u32],
        prepared: Option<&P2pPreparedRequest>,
        model_id: Option<&str>,
    ) -> P2pFreshPlan<String> {
        let Some(selector) = self.p2p_selector.as_ref() else {
            return P2pFreshPlan::Stop {
                reason: "p2p_selector_unavailable",
                fallback: None,
            };
        };
        let Some(prepared) = prepared else {
            return P2pFreshPlan::Stop {
                reason: "p2p_request_hashes_unavailable",
                fallback: None,
            };
        };
        let workers = self.available_p2p_prefills(model_id);
        let Some(source_match) =
            selector.match_source_prepared(&workers, request_tokens.len(), prepared)
        else {
            return P2pFreshPlan::Stop {
                reason: "fresh_kv_owner_unavailable",
                fallback: None,
            };
        };
        let Some(source) = workers.get(source_match.source_index).cloned() else {
            return P2pFreshPlan::Stop {
                reason: "fresh_kv_owner_index_invalid",
                fallback: None,
            };
        };
        if source_match.source_bootstrap_addr.is_none() {
            return P2pFreshPlan::Stop {
                reason: "fresh_kv_owner_missing_bootstrap",
                fallback: Some(source),
            };
        }

        let candidates = workers
            .into_iter()
            .filter(|target| {
                target.is_available() && selector.is_distinct_node(source.as_ref(), target.as_ref())
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return P2pFreshPlan::Stop {
                reason: "fresh_p2p_target_unavailable",
                fallback: Some(source),
            };
        }
        if !candidates
            .iter()
            .any(|target| selector.pair_is_beneficial(source.as_ref(), target.as_ref()))
        {
            return P2pFreshPlan::Stop {
                reason: "fresh_pair_no_longer_beneficial",
                fallback: Some(source),
            };
        }

        P2pFreshPlan::Candidate {
            context: source.url().to_string(),
            source,
            candidates,
        }
    }

    fn validate_granted_p2p_target(
        &self,
        request_tokens: &[u32],
        prepared: Option<&P2pPreparedRequest>,
        model_id: Option<&str>,
        expected_source_url: &str,
        target: &Arc<dyn Worker>,
    ) -> Option<RemoteKvDecision> {
        let selector = self.p2p_selector.as_ref()?;
        let prepared = prepared?;
        let workers = self.available_p2p_prefills(model_id);
        let selection = selector.select_for_target_prepared(
            &workers,
            request_tokens,
            prepared,
            target.url(),
        )?;
        let decision = selection.remote_kv?;
        let source_matches =
            decision.source_url.trim_end_matches('/') == expected_source_url.trim_end_matches('/');
        let target_matches =
            decision.target_url.trim_end_matches('/') == target.url().trim_end_matches('/');
        (source_matches && target_matches).then_some(decision)
    }

    async fn select_pd_pair(
        &self,
        request_text: Option<&str>,
        model_id: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        let selected = self
            .select_pd_pair_with_decision(request_text, None, model_id, headers)
            .await?;
        Ok((selected.prefill, selected.decode))
    }

    async fn select_pd_pair_with_decision(
        &self,
        request_text: Option<&str>,
        request_tokens: Option<&[u32]>,
        model_id: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Result<SelectedPdPair, String> {
        let effective_model_id = if !self.enable_igw { None } else { model_id };

        debug!(
            "Selecting PD pair: enable_igw={}, model_id={:?}, effective_model_id={:?}",
            self.enable_igw, model_id, effective_model_id
        );

        let prefill_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_prefill_workers()
        };

        let decode_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Decode))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_decode_workers()
        };

        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        // Get cached hash ring for consistent hashing
        let hash_ring = self
            .worker_registry
            .get_hash_ring(effective_model_id.unwrap_or(UNKNOWN_MODEL_ID));

        let mut remote_kv = None;
        let mut prepared_p2p_request = None;
        let prefill =
            if let (Some(selector), Some(tokens)) = (self.p2p_selector.as_ref(), request_tokens) {
                let available_prefills: Vec<_> = prefill_workers
                    .iter()
                    .filter(|worker| worker.is_available())
                    .cloned()
                    .collect();
                let prepared = selector.prepare_request(tokens);
                let selection = prepared.as_ref().and_then(|prepared| {
                    selector.select_prepared(&available_prefills, tokens, prepared)
                });
                prepared_p2p_request = prepared;
                match selection {
                    Some(selection) => {
                        let selected = available_prefills
                            .get(selection.target_index)
                            .cloned()
                            .ok_or_else(|| {
                                "P2P selector returned an invalid Prefill index".to_string()
                            })?;
                        remote_kv = selection.remote_kv;
                        selected
                    }
                    None => {
                        Self::pick_worker_by_policy_arc(
                            &prefill_workers,
                            &*prefill_policy,
                            request_text,
                            headers,
                            hash_ring.clone(),
                            "prefill",
                        )
                        .await?
                    }
                }
            } else {
                Self::pick_worker_by_policy_arc(
                    &prefill_workers,
                    &*prefill_policy,
                    request_text,
                    headers,
                    hash_ring.clone(),
                    "prefill",
                )
                .await?
            };

        let decode = Self::pick_worker_by_policy_arc(
            &decode_workers,
            &*decode_policy,
            request_text,
            headers,
            hash_ring,
            "decode",
        )
        .await?;

        // Record worker selection metrics (Layer 3)
        let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_HTTP,
            model,
            prefill_policy.name(),
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            metrics_labels::CONNECTION_HTTP,
            model,
            decode_policy.name(),
        );

        Ok(SelectedPdPair {
            prefill,
            decode,
            remote_kv,
            prepared_p2p_request,
        })
    }

    async fn pick_worker_by_policy_arc(
        workers: &[Arc<dyn Worker>],
        policy: &dyn LoadBalancingPolicy,
        request_text: Option<&str>,
        headers: Option<&HeaderMap>,
        hash_ring: Option<Arc<HashRing>>,
        worker_type: &str,
    ) -> Result<Arc<dyn Worker>, String> {
        if workers.is_empty() {
            return Err(format!(
                "No {} workers available. Please check if {} servers are configured and healthy.",
                worker_type, worker_type
            ));
        }

        let available_workers: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available_workers.is_empty() {
            return Err(format!(
                "No available {} workers (all circuits open or unhealthy)",
                worker_type
            ));
        }

        let selected_idx = policy
            .select_worker(
                &available_workers,
                &SelectWorkerInfo {
                    request_text,
                    tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                    headers,
                    hash_ring,
                },
            )
            .await
            .ok_or_else(|| {
                format!(
                    "Policy {} failed to select a {} worker",
                    policy.name(),
                    worker_type
                )
            })?;

        Ok(available_workers[selected_idx].clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn create_streaming_response(
        &self,
        stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
        status: StatusCode,
        prefill_logprobs: Option<Value>,
        return_logprob: bool,
        headers: Option<HeaderMap>,
        decode: Arc<dyn Worker>,
        guards: Vec<WorkerLoadGuard>,
    ) -> Response {
        use crate::core::AttachedBody;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // Uses select! to race stream.next() against tx.closed() so that
        // when the client disconnects the upstream HTTP connection is dropped
        // promptly, allowing the engine to abort the request.
        // `biased;` drains a ready upstream chunk before observing client
        // disconnect, so a chunk already produced by reqwest reaches the
        // client (and the logprob merger) before we tear the loop down.
        //
        // The upstream stream is wrapped in `BreakerTrackedStream` so the
        // decode worker's circuit breaker is updated once on drop: success
        // on clean completion (`[DONE]` sentinel or `None`), failure on
        // stream error, neither on client disconnect. PD's pre-PR semantics
        // treated 4xx (client error) as not-a-worker-fault, so we only
        // pre-mark the wrapper as Errored on 5xx — `handle_decode_error_response`
        // synthesizes a single-chunk SSE error envelope that would otherwise
        // stream cleanly to None and record a spurious success.
        let mut tracked =
            BreakerTrackedStream::new(stream, Arc::clone(&decode), decode.url().to_string());
        if !(status.is_success() || status.is_client_error()) {
            tracked.mark_errored();
        }
        let decode_for_log = decode.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    chunk_result = tracked.next() => {
                        match chunk_result {
                            Some(Ok(chunk)) => {
                                let is_done = memmem::find(&chunk, b"data: [DONE]").is_some();

                                let result = if return_logprob && prefill_logprobs.is_some() {
                                    Self::merge_streaming_logprobs(prefill_logprobs.clone(), &chunk)
                                        .unwrap_or(chunk)
                                } else {
                                    chunk
                                };

                                // Mark the wrapper completed before the client
                                // send: upstream finished cleanly regardless of
                                // whether the client is still listening, and
                                // the worker deserves the success tick either
                                // way. `mark_completed` is a no-op once Errored
                                // is set, so the synthetic-error path is unaffected.
                                if is_done {
                                    tracked.mark_completed();
                                }

                                if tx.send(Ok(result)).is_err() {
                                    tracing::debug!(
                                        "Receiver dropped (likely client disconnect), \
                                        cancelling upstream PD stream"
                                    );
                                    break;
                                }

                                if is_done {
                                    break;
                                }
                            }
                            Some(Err(e)) => {
                                // BreakerTrackedStream already logged the error
                                // and marked the terminal state as Errored so
                                // the worker's circuit breaker will tick on drop.
                                let _ = tx.send(Err(format!("Stream error: {}", e)));
                                break;
                            }
                            None => break,
                        }
                    }
                    _ = tx.closed() => {
                        tracing::info!(
                            "Client disconnected, cancelling upstream PD stream from {}",
                            decode_for_log.url()
                        );
                        break;
                    }
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        let body = Body::from_stream(stream);

        let mut response = Response::new(body);
        *response.status_mut() = status;

        let mut response_headers = headers.unwrap_or_default();
        response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        *response.headers_mut() = response_headers;

        AttachedBody::wrap_response(response, guards)
    }

    // Helper to process non-streaming decode response with logprob merging
    async fn process_non_streaming_response(
        &self,
        res: reqwest::Response,
        status: StatusCode,
        return_logprob: bool,
        prefill_body: Option<bytes::Bytes>,
    ) -> Response {
        let response = res.bytes().await;
        let decode_body = match response {
            Ok(decode_body) => decode_body,
            Err(e) => {
                error!("Failed to read decode response: {}", e);
                return error::internal_error("read_response_failed", "Failed to read response");
            }
        };

        if !return_logprob {
            return (status, decode_body).into_response();
        }

        let Some(prefill_body) = prefill_body else {
            return (status, decode_body).into_response();
        };

        // Merge logprobs from prefill and decode
        let (Ok(prefill_json), Ok(mut decode_json)) = (
            serde_json::from_slice::<Value>(&prefill_body),
            serde_json::from_slice::<Value>(&decode_body),
        ) else {
            warn!("Failed to parse responses for logprob merging");
            return (status, decode_body).into_response();
        };

        Self::merge_logprobs_in_json(&prefill_json, &mut decode_json);

        // Return merged response
        match serde_json::to_vec(&decode_json) {
            Ok(body) => (status, body).into_response(),
            Err(e) => {
                error!("Failed to serialize merged response: {}", e);
                (status, decode_body).into_response()
            }
        }
    }

    // Helper to process prefill response and extract body if needed for logprobs
    async fn process_prefill_response(
        prefill_result: Result<reqwest::Response, reqwest::Error>,
        prefill_url: &str,
        return_logprob: bool,
    ) -> Result<(StatusCode, Option<bytes::Bytes>), Response> {
        // Check prefill result first - it's critical for disaggregated mode
        let prefill_response = match prefill_result {
            Ok(response) => response,
            Err(e) => {
                error!(
                    "Prefill server failed (CRITICAL) prefill_url={} error={}. Decode will timeout without prefill KV cache.",
                    prefill_url,
                    e
                );

                // Return error immediately - don't wait for decode to timeout
                return Err(error::bad_gateway(
                    "prefill_server_error",
                    format!(
                        "Prefill server error: {}. This will cause decode timeout.",
                        e
                    ),
                ));
            }
        };

        let prefill_status = StatusCode::from_u16(prefill_response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // Check if prefill succeeded
        if !prefill_status.is_success() {
            // Get error body from prefill
            let error_msg = prefill_response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown prefill error".to_string());

            error!(
                "Prefill server returned error status prefill_url={} status={} body={}",
                prefill_url, prefill_status, error_msg
            );

            // Map prefill_status to appropriate error function
            let error_response = match prefill_status {
                StatusCode::BAD_REQUEST => error::bad_request(
                    "prefill_bad_request",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::NOT_FOUND => error::not_found(
                    "prefill_not_found",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                    "prefill_internal_error",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                    "prefill_unavailable",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::BAD_GATEWAY => error::bad_gateway(
                    "prefill_bad_gateway",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                _ => error::internal_error(
                    "prefill_error",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
            };
            return Err(error_response);
        }

        // Read prefill body if needed for logprob merging
        let prefill_body = if return_logprob {
            match prefill_response.bytes().await {
                Ok(body) => Some(body),
                Err(e) => {
                    warn!("Failed to read prefill response body for logprobs: {}", e);
                    None
                }
            }
        } else {
            // For non-logprob requests, just consume the response without storing
            debug!("Consuming prefill response body (non-logprob request)");
            match prefill_response.bytes().await {
                Ok(_) => debug!("Prefill response consumed successfully"),
                Err(e) => warn!("Error consuming prefill response: {}", e),
            }
            None
        };

        Ok((prefill_status, prefill_body))
    }

    fn build_post_with_headers(
        &self,
        client: &Client,
        url: &str,
        route: &'static str,
        json_request: &Value,
        headers: Option<&HeaderMap>,
        connection_close: bool,
    ) -> reqwest::RequestBuilder {
        let mut request = client.post(api_path(url, route)).json(json_request);
        if connection_close {
            request = request.header("Connection", "close");
        }
        if let Some(headers) = headers {
            for (name, value) in headers.iter() {
                if header_utils::should_forward_request_header(name.as_str()) {
                    if let Ok(val) = value.to_str() {
                        request = request.header(name, val);
                    }
                }
            }
        }
        request
    }

    // Helper to merge logprobs from prefill and decode responses
    // Optimized to avoid double cloning by taking ownership of decode array
    fn merge_logprobs_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
        if let (Some(prefill_meta), Some(decode_meta)) = (
            prefill_json.get("meta_info"),
            decode_json.get_mut("meta_info"),
        ) {
            if let (Some(prefill_logprobs), Some(decode_logprobs)) = (
                prefill_meta.get("input_token_logprobs"),
                decode_meta.get_mut("input_token_logprobs"),
            ) {
                if let Some(prefill_arr) = prefill_logprobs.as_array() {
                    // Take ownership of decode array to avoid cloning it
                    let decode_arr = std::mem::take(decode_logprobs);
                    if let Value::Array(decode_vec) = decode_arr {
                        // Pre-allocate merged array with exact capacity
                        let mut merged = Vec::with_capacity(prefill_arr.len() + decode_vec.len());
                        merged.extend(prefill_arr.iter().cloned());
                        merged.extend(decode_vec);
                        decode_meta["input_token_logprobs"] = Value::Array(merged);
                        return true;
                    }
                }
            }
        }
        false
    }

    // Simple helper to merge logprobs in streaming responses
    // Optimized to reduce allocations in the merge path
    fn merge_streaming_logprobs(
        prefill_logprobs: Option<Value>,
        decode_chunk: &[u8],
    ) -> Result<bytes::Bytes, ()> {
        // Skip non-data chunks
        let chunk_str = std::str::from_utf8(decode_chunk).map_err(|_| ())?;
        if !chunk_str.starts_with("data: ") || chunk_str.contains("[DONE]") {
            return Err(());
        }

        // Parse JSON from chunk
        let json_str = chunk_str.trim_start_matches("data: ").trim();
        let mut decode_json: Value = serde_json::from_str(json_str).map_err(|_| ())?;

        // Merge prefill logprobs if available
        if let Some(ref p_logprobs) = prefill_logprobs {
            if let Some(meta) = decode_json.get_mut("meta_info") {
                if let Some(d_logprobs) = meta.get_mut("input_token_logprobs") {
                    if let Some(p_arr) = p_logprobs.as_array() {
                        // Take ownership of decode array to avoid cloning it
                        let decode_arr = std::mem::take(d_logprobs);
                        if let Value::Array(d_vec) = decode_arr {
                            // Pre-allocate merged array with exact capacity
                            let mut merged = Vec::with_capacity(p_arr.len() + d_vec.len());
                            merged.extend(p_arr.iter().cloned());
                            merged.extend(d_vec);
                            *d_logprobs = Value::Array(merged);
                        }
                    }
                }
            }
        }

        // Re-serialize
        let merged_str = format!(
            "data: {}\n\n",
            serde_json::to_string(&decode_json).unwrap_or_default()
        );
        Ok(bytes::Bytes::from(merged_str))
    }
}

#[async_trait]
impl RouterTrait for PDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Note: This endpoint actually causes the model to generate tokens, so we only test one pair

        // Select a random worker pair using the policy
        let (prefill, decode) = match self.select_pd_pair(None, None, None).await {
            Ok(pair) => pair,
            Err(e) => {
                return error::service_unavailable(
                    "no_healthy_worker_pair",
                    format!("No healthy worker pair available: {}", e),
                );
            }
        };

        let prefill_url = format!("{}/health_generate", prefill.url());
        let (prefill_result, decode_result) = tokio::join!(
            self.client.get(&prefill_url).send(),
            self.client
                .get(format!("{}/health_generate", decode.url()))
                .send()
        );

        // Check results
        let mut errors = Vec::new();

        match prefill_result {
            Ok(res) if res.status().is_success() => {
                debug!(
                    "Health generate passed for prefill server: {}",
                    prefill.url()
                );
            }
            Ok(res) => {
                errors.push(format!(
                    "Prefill {} returned status {}",
                    prefill.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            }
        }

        match decode_result {
            Ok(res) if res.status().is_success() => {
                debug!("Health generate passed for decode server: {}", decode.url());
            }
            Ok(res) => {
                errors.push(format!(
                    "Decode {} returned status {}",
                    decode.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Decode {} error: {}", decode.url(), e));
            }
        }

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!(
                    "Health generate passed on selected pair: prefill={}, decode={}",
                    prefill.url(),
                    decode.url()
                ),
            )
                .into_response()
        } else {
            error::service_unavailable(
                "health_generate_failed",
                format!("Health generate failed: {:?}", errors),
            )
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // Get info from the first decode server to match sglang's server info format
        // Note: We use decode workers for server info to match expected format
        self.proxy_to_first_prefill_worker("server_info", None)
            .await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("v1/models", Some(headers))
            .await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("model_info", Some(headers))
            .await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.return_logprob.unwrap_or(false);

        let request_text = if self.policies_need_request_text() {
            body.text.as_deref().map(|s| s.to_string())
        } else {
            None
        };

        let batch_size = Self::get_generate_batch_size(body);
        let request_tokens = self
            .p2p_selector
            .as_ref()
            .and_then(|_| self.p2p_tokens_for_generate(body, model_id));

        let context = PDRequestContext {
            route: "/generate",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            request_tokens,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_chat_with_input_ids(headers, body, model_id, None)
            .await
    }

    async fn route_chat_with_input_ids(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
        input_ids: Option<&InputIds>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs;
        let upstream_tokens = input_ids
            .and_then(Self::p2p_tokens_from_input_ids)
            .filter(|tokens| !tokens.is_empty());
        let header_prompt_tokens = headers
            .and_then(|headers| headers.get("x-sglang-prompt-tokens"))
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<usize>().ok());
        let original_path = headers
            .and_then(|headers| headers.get("x-envoy-original-path"))
            .and_then(|value| value.to_str().ok());
        if input_ids.is_some() && upstream_tokens.is_none() {
            warn!(
                ?header_prompt_tokens,
                ?original_path,
                "Invalid chat input_ids ignored for both P2P routing and backend forwarding"
            );
        } else if input_ids.is_none() && header_prompt_tokens.is_some() {
            warn!(
                ?header_prompt_tokens,
                ?original_path,
                "Pretokenized prompt length arrived without chat input_ids"
            );
        } else if let (Some(header_count), Some(tokens)) =
            (header_prompt_tokens, upstream_tokens.as_ref())
        {
            if header_count != tokens.len() {
                warn!(
                    header_prompt_tokens = header_count,
                    input_ids_count = tokens.len(),
                    ?original_path,
                    "Pretokenized prompt length does not match chat input_ids"
                );
            }
        }

        let request_text = if self.policies_need_request_text() {
            body.messages.first().and_then(|msg| match msg {
                ChatMessage::User { content, .. } => match content {
                    MessageContent::Text(text) => Some(text.clone()),
                    MessageContent::Parts(_) => None,
                },
                ChatMessage::Developer { content, .. } => match content {
                    MessageContent::Text(text) => Some(text.clone()),
                    MessageContent::Parts(_) => None,
                },
                ChatMessage::System { content, .. } => Some(content.to_simple_string()),
                _ => None,
            })
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_chat_batch_size(body);
        let request_tokens = if upstream_tokens.is_some() {
            info!(
                token_source = "upstream",
                token_count = upstream_tokens.as_ref().map_or(0, Vec::len),
                "P2P routing token source selected"
            );
            upstream_tokens.clone()
        } else {
            self.p2p_selector
                .as_ref()
                .and_then(|_| self.p2p_tokens_for_chat(body, model_id, None))
        };

        let context = PDRequestContext {
            route: "/v1/chat/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            request_tokens: request_tokens.clone(),
            model_id,
            headers: headers.cloned(),
        };

        let forwarded = ForwardedChatRequest {
            request: body,
            input_ids: request_tokens.as_deref(),
        };
        self.execute_dual_dispatch(headers, &forwarded, context)
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs.is_some();

        let request_text = if self.policies_need_request_text() {
            match &body.prompt {
                StringOrArray::String(s) => Some(s.clone()),
                StringOrArray::Array(v) => v.first().map(|s| s.to_string()),
            }
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_completion_batch_size(body);
        let request_tokens = self
            .p2p_selector
            .as_ref()
            .and_then(|_| self.p2p_tokens_for_completion(body, model_id));

        let context = PDRequestContext {
            route: "/v1/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            request_tokens,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Extract text for cache-aware routing
        let req_text = if self.policies_need_request_text() {
            Some(body.query.clone())
        } else {
            None
        };

        let context = PDRequestContext {
            route: "/v1/rerank",
            batch_size: None,
            is_stream: false,
            return_logprob: false,
            request_text: req_text,
            request_tokens: None,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        let _ = (headers, body, model_id);
        warn!("PD mode does not support /v1/embeddings; returning bad request");
        error::bad_request(
            "pd_unsupported_embeddings",
            "PD mode does not support /v1/embeddings",
        )
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        let _ = (headers, body, model_id);
        warn!("PD mode does not support /v1/classify; returning bad request");
        error::bad_request(
            "pd_unsupported_classify",
            "PD mode does not support /v1/classify",
        )
    }

    fn router_type(&self) -> &'static str {
        "pd"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};
    use axum::{routing::post, Json, Router};
    use tokio::net::TcpListener;

    fn load_test_truncated_tokenizer() -> (tokenizers::Tokenizer, Option<usize>) {
        let directory = tempfile::tempdir().unwrap();
        let tokenizer_path = directory.path().join("tokenizer.json");
        std::fs::write(
            &tokenizer_path,
            serde_json::to_vec(&json!({
                "version": "1.0",
                "truncation": {
                    "direction": "Right",
                    "max_length": 4,
                    "strategy": "LongestFirst",
                    "stride": 0
                },
                "padding": null,
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": null,
                "decoder": null,
                "model": {
                    "type": "WordLevel",
                    "vocab": {"[UNK]": 0, "x": 1},
                    "unk_token": "[UNK]"
                }
            }))
            .unwrap(),
        )
        .unwrap();
        PDRouter::load_p2p_untruncated_tokenizer(directory.path().to_str().unwrap()).unwrap()
    }

    fn create_test_pd_router() -> PDRouter {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));

        PDRouter {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            api_key: Some("test_api_key".to_string()),
            enable_igw: false,
            tokenizer_registry: Arc::new(TokenizerRegistry::new()),
            p2p_untruncated_tokenizer: None,
            p2p_selector: None,
            p2p_node_gate: None,
        }
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .build();
        worker.set_healthy(healthy);
        Box::new(worker)
    }

    fn test_p2p_decision(target_url: String) -> RemoteKvDecision {
        RemoteKvDecision {
            source_url: "http://source".to_string(),
            source_bootstrap_addr: "source:8998".to_string(),
            target_url,
            matched_tokens: 3,
            token_ids: vec![1, 2, 3],
            reason: "load_imbalance",
        }
    }

    async fn spawn_p2p_control_worker(response: Value) -> (String, tokio::task::JoinHandle<()>) {
        let app = Router::new().route(
            "/experimental/p2p_kv_transfer",
            post(move |Json(_request): Json<Value>| {
                let response = response.clone();
                async move { Json(response) }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), task)
    }

    async fn spawn_blocked_p2p_control_worker(
        response: Value,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let app = Router::new().route(
            "/experimental/p2p_kv_transfer",
            post(move |Json(_request): Json<Value>| {
                let response = response.clone();
                let started = Arc::clone(&started);
                let release = Arc::clone(&release);
                async move {
                    started.notify_one();
                    release.notified().await;
                    Json(response)
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), task)
    }

    #[tokio::test]
    async fn abnormal_transfer_lease_drop_releases_only_its_nodes() {
        let gate = P2pNodeGate::new_isolated(Duration::from_millis(20), 0);
        let node_lease = gate
            .acquire("http://worker-a", "http://worker-b")
            .await
            .expect("lease must enter");
        let lease = P2pTransferLease {
            _node_lease: node_lease,
            explicitly_settled: false,
            acquired_at: Instant::now(),
            source_url: "http://worker-a".to_string(),
            target_url: "http://worker-b".to_string(),
            matched_tokens: 1,
            attempt_id: "router-p2p-abnormal-drop".to_string(),
        };

        drop(lease);
        assert!(
            gate.acquire("http://worker-a", "http://worker-b")
                .await
                .is_some(),
            "an abnormal Router task must not permanently blacklist either worker"
        );
    }

    #[test]
    fn independent_p2p_payload_is_not_forwarded_to_normal_pd() {
        let decision = test_p2p_decision("http://target".to_string());
        let control = PDRouter::p2p_transfer_payload(&decision, "router-p2p-123");
        assert_eq!(control["source_url"], "http://source");
        assert_eq!(control["target_url"], "http://target");
        assert_eq!(control["source_bootstrap_addr"], "source:8998");
        assert_eq!(control["matched_tokens"], 3);
        assert_eq!(control["token_ids"], json!([1, 2, 3]));
        assert_eq!(control["request_id"], "router-p2p-123");
        assert_eq!(control["dry_run"], false);

        let request = json!({
            "model": "glm",
            "prompt": "hello",
            "remote_kv_source_url": "http://source",
            "remote_kv_source_bootstrap_addr": "source:8998",
            "remote_kv_target_url": "http://target",
            "remote_kv_matched_tokens": 160_000,
            "remote_kv_token_ids": [1, 2, 3],
            "remote_kv_reason": "load_imbalance",
            "remote_kv_attempt_id": "untrusted-inbound-attempt"
        });
        let (prefill, decode) = PDRouter::prepare_pd_payloads(&request).unwrap();

        for payload in [&prefill, &decode] {
            for key in PDRouter::REMOTE_KV_KEYS {
                assert!(
                    payload.get(key).is_none(),
                    "fallback payload retained remote KV field {key}"
                );
            }
            assert_eq!(payload["model"], "glm");
            assert_eq!(payload["prompt"], "hello");
        }

        let mut inbound = HeaderMap::new();
        inbound.insert("authorization", HeaderValue::from_static("Bearer test"));
        inbound.insert(
            "x-sgl-remote-kv-attempt-id",
            HeaderValue::from_static("untrusted-inbound-attempt"),
        );
        let (prefill_headers, decode_headers) = PDRouter::prepare_pd_headers(&inbound);
        for headers in [&prefill_headers, &decode_headers] {
            for key in PDRouter::REMOTE_KV_HEADER_KEYS {
                assert!(
                    headers.get(key).is_none(),
                    "normal PD headers retained remote KV hint {key}"
                );
            }
            assert_eq!(headers.get("authorization").unwrap(), "Bearer test");
        }
    }

    #[test]
    fn namespaced_or_lora_requests_fail_closed_for_p2p() {
        for key in PDRouter::P2P_CACHE_NAMESPACE_KEYS {
            let request = Value::Object(serde_json::Map::from_iter([(
                key.to_string(),
                json!("non-empty"),
            )]));
            assert_eq!(
                PDRouter::p2p_nonempty_cache_namespace(&request),
                Some(key),
                "{key} must disable P2P until the control payload reproduces its cache namespace"
            );
        }
        assert_eq!(
            PDRouter::p2p_nonempty_cache_namespace(&json!({
                "extra_key": "",
                "cache_salt": null,
                "lora_id": "",
                "lora_path": null,
            })),
            None
        );
    }

    #[test]
    fn p2p_response_endpoints_are_checked_when_worker_returns_them() {
        assert!(PDRouter::p2p_response_endpoint_matches(
            &json!({}),
            "source_url",
            "http://source"
        ));
        assert!(PDRouter::p2p_response_endpoint_matches(
            &json!({"source_url": "http://source/"}),
            "source_url",
            "http://source"
        ));
        assert!(!PDRouter::p2p_response_endpoint_matches(
            &json!({"source_url": "http://different"}),
            "source_url",
            "http://source"
        ));
    }

    #[tokio::test]
    async fn terminal_p2p_control_releases_node_locks_before_normal_prefill() {
        let (target_url, server) = spawn_p2p_control_worker(json!({
            "success": true,
            "message": "transferred",
            "matched_tokens": 3,
            "transferred_tokens": 3,
            "fallback_recompute": false,
        }))
        .await;
        let decision = test_p2p_decision(format!("{target_url}/"));
        let gate = P2pNodeGate::new_isolated(Duration::from_millis(20), 0);
        let node_lease = gate
            .acquire(&decision.source_url, &decision.target_url)
            .await
            .expect("lease must enter");
        let lease = P2pTransferLease {
            _node_lease: node_lease,
            explicitly_settled: false,
            acquired_at: Instant::now(),
            source_url: decision.source_url.clone(),
            target_url: decision.target_url.clone(),
            matched_tokens: decision.matched_tokens,
            attempt_id: "router-p2p-terminal".to_string(),
        };

        let router = create_test_pd_router();
        let outcome = router
            .execute_independent_p2p_transfer(&HeaderMap::new(), &decision, lease)
            .await;
        assert_eq!(
            outcome,
            P2pTransferOutcome::Transferred {
                transferred_tokens: 3
            }
        );
        assert!(
            gate.acquire(&decision.source_url, &decision.target_url)
                .await
                .is_some(),
            "normal Prefill must start after both P2P node locks are released"
        );
        server.abort();
    }

    #[tokio::test]
    async fn cancelling_handler_does_not_drop_active_p2p_node_locks() {
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let (target_url, server) = spawn_blocked_p2p_control_worker(
            json!({
                "success": true,
                "message": "transferred",
                "matched_tokens": 3,
                "transferred_tokens": 3,
                "fallback_recompute": false,
            }),
            Arc::clone(&started),
            Arc::clone(&release),
        )
        .await;
        let decision = test_p2p_decision(target_url);
        let gate = P2pNodeGate::new_isolated(Duration::from_millis(20), 0);
        let node_lease = gate
            .acquire(&decision.source_url, &decision.target_url)
            .await
            .expect("lease must enter");
        let lease = P2pTransferLease {
            _node_lease: node_lease,
            explicitly_settled: false,
            acquired_at: Instant::now(),
            source_url: decision.source_url.clone(),
            target_url: decision.target_url.clone(),
            matched_tokens: decision.matched_tokens,
            attempt_id: "router-p2p-cancelled-handler".to_string(),
        };

        let router = Arc::new(create_test_pd_router());
        let control_task = tokio::spawn({
            let router = Arc::clone(&router);
            let decision = decision.clone();
            async move {
                router
                    .execute_independent_p2p_transfer(&HeaderMap::new(), &decision, lease)
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(1), started.notified())
            .await
            .expect("control request must reach the worker");

        control_task.abort();
        assert!(control_task.await.unwrap_err().is_cancelled());
        assert!(
            gate.acquire(&decision.source_url, &decision.target_url)
                .await
                .is_none(),
            "detached control settlement must retain both locks after handler cancellation"
        );

        release.notify_waiters();
        let reacquired = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if let Some(lease) = gate
                    .acquire(&decision.source_url, &decision.target_url)
                    .await
                {
                    break lease;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("detached settlement must eventually release both locks");
        drop(reacquired);
        server.abort();
    }

    #[tokio::test]
    async fn invalid_p2p_success_claim_falls_back_and_releases_node_locks() {
        let (target_url, server) = spawn_p2p_control_worker(json!({
            "success": true,
            "message": "invalid oversized transfer",
            "source_url": "http://source",
            "matched_tokens": 3,
            "transferred_tokens": 4,
            "fallback_recompute": false,
        }))
        .await;
        let decision = test_p2p_decision(target_url);
        let gate = P2pNodeGate::new_isolated(Duration::from_millis(20), 0);
        let node_lease = gate
            .acquire(&decision.source_url, &decision.target_url)
            .await
            .expect("lease must enter");
        let lease = P2pTransferLease {
            _node_lease: node_lease,
            explicitly_settled: false,
            acquired_at: Instant::now(),
            source_url: decision.source_url.clone(),
            target_url: decision.target_url.clone(),
            matched_tokens: decision.matched_tokens,
            attempt_id: "router-p2p-invalid-success".to_string(),
        };

        let outcome = create_test_pd_router()
            .execute_independent_p2p_transfer(&HeaderMap::new(), &decision, lease)
            .await;
        assert_eq!(outcome, P2pTransferOutcome::Fallback);
        assert!(gate
            .acquire(&decision.source_url, &decision.target_url)
            .await
            .is_some());
        server.abort();
    }

    #[tokio::test]
    async fn uncertain_p2p_transport_releases_router_locks_for_local_recompute() {
        let decision = test_p2p_decision("not a valid worker URL".to_string());
        let gate = P2pNodeGate::new_isolated(Duration::from_millis(20), 0);
        let node_lease = gate
            .acquire(&decision.source_url, &decision.target_url)
            .await
            .expect("lease must enter");
        let lease = P2pTransferLease {
            _node_lease: node_lease,
            explicitly_settled: false,
            acquired_at: Instant::now(),
            source_url: decision.source_url.clone(),
            target_url: decision.target_url.clone(),
            matched_tokens: decision.matched_tokens,
            attempt_id: "router-p2p-uncertain".to_string(),
        };

        let router = create_test_pd_router();
        let outcome = router
            .execute_independent_p2p_transfer(&HeaderMap::new(), &decision, lease)
            .await;
        assert_eq!(outcome, P2pTransferOutcome::TransportUncertain);
        assert!(
            gate.acquire(&decision.source_url, &decision.target_url)
                .await
                .is_some(),
            "Router locks must not become a permanent blacklist after an uncertain transport"
        );
    }

    #[tokio::test]
    async fn test_select_healthy_prefill_worker() {
        let router = create_test_pd_router();

        let healthy_worker = create_test_worker(
            "http://healthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let unhealthy_worker = create_test_worker(
            "http://unhealthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            false,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(unhealthy_worker));
        router.worker_registry.register(Arc::from(healthy_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_ok());
        let (prefill, _decode) = result.unwrap();

        assert_eq!(prefill.url(), "http://healthy");
        assert!(prefill.is_healthy());
    }

    #[tokio::test]
    async fn test_empty_worker_lists() {
        let router = create_test_pd_router();

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No prefill workers available"));
    }

    #[test]
    fn test_worker_load_metrics() {
        let prefill_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        ));
        let decode_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".to_string(),
            WorkerType::Decode,
            true,
        ));

        let _prefill_guard = WorkerLoadGuard::new(prefill_worker.clone(), None);
        let _decode_guard = WorkerLoadGuard::new(decode_worker.clone(), None);

        assert_eq!(prefill_worker.load(), 1);
        assert_eq!(decode_worker.load(), 1);

        drop(_prefill_guard);
        drop(_decode_guard);

        assert_eq!(prefill_worker.load(), 0);
        assert_eq!(decode_worker.load(), 0);
    }

    #[test]
    fn upstream_chat_input_ids_bypass_local_tokenizer() {
        let router = create_test_pd_router();
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "glm",
            "messages": [{"role": "user", "content": "hello"}],
        }))
        .unwrap();
        let ids = InputIds::Single((0..40_001).map(|id| id % 128_000).collect());
        let normalized_ids =
            PDRouter::p2p_tokens_from_input_ids(&ids).expect("flat IDs should normalize");

        let tokens = router
            .p2p_tokens_for_chat(&request, Some("glm"), Some(&normalized_ids))
            .expect("valid upstream input_ids should not need a local tokenizer");

        assert_eq!(tokens.len(), 40_001);
        assert_eq!(tokens[0], 0);
        assert_eq!(tokens[40_000], 40_000);
    }

    #[test]
    fn upstream_chat_input_ids_are_forwarded_without_truncation() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "glm",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
        }))
        .unwrap();
        let ids = InputIds::Single((0..100_001).map(|id| id % 128_000).collect());
        let normalized_ids =
            PDRouter::p2p_tokens_from_input_ids(&ids).expect("flat IDs should normalize");
        let forwarded = ForwardedChatRequest {
            request: &request,
            input_ids: Some(&normalized_ids),
        };

        let value = serde_json::to_value(forwarded).unwrap();
        assert_eq!(value["input_ids"].as_array().map(Vec::len), Some(100_001));
        assert_eq!(value["input_ids"][100_000], 100_000);
        assert_eq!(
            value["messages"],
            json!([{"role": "user", "content": "hello"}])
        );
        assert_eq!(value["stream"], true);
    }

    #[test]
    fn p2p_tokenizer_disables_embedded_truncation() {
        let (tokenizer, previous_max_length) = load_test_truncated_tokenizer();
        assert_eq!(previous_max_length, Some(4));
        assert!(tokenizer.get_truncation().is_none());
        assert_eq!(
            tokenizer.encode("x ".repeat(100), false).unwrap().len(),
            100
        );
    }

    #[tokio::test]
    async fn igw_matches_untruncated_tokenizer_by_registered_source() {
        use crate::tokenizer::{MockTokenizer, TokenizerTrait};

        let mut router = create_test_pd_router();
        router.enable_igw = true;
        router
            .tokenizer_registry
            .load("glm-tokenizer", "glm-served", "/models/glm", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn TokenizerTrait>)
            })
            .await
            .unwrap();
        router
            .tokenizer_registry
            .load(
                "other-tokenizer",
                "other-served",
                "/models/other",
                || async { Ok(Arc::new(MockTokenizer::default()) as Arc<dyn TokenizerTrait>) },
            )
            .await
            .unwrap();
        let (tokenizer, _) = load_test_truncated_tokenizer();
        router.p2p_untruncated_tokenizer = Some(P2pUntruncatedTokenizer {
            source: "/models/glm".to_string(),
            tokenizer: Arc::new(tokenizer),
        });

        assert!(router
            .p2p_untruncated_tokenizer_for_model("glm-served")
            .is_some());
        assert!(router
            .p2p_untruncated_tokenizer_for_model("other-served")
            .is_none());
    }

    #[test]
    fn invalid_chat_input_ids_fall_back_to_local_tokenizer() {
        let router = create_test_pd_router();
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "glm",
            "messages": [{"role": "user", "content": "hello"}],
        }))
        .unwrap();

        let negative = InputIds::Single(vec![1, -1, 2]);
        assert!(router
            .p2p_tokens_for_chat(
                &request,
                Some("glm"),
                PDRouter::p2p_tokens_from_input_ids(&negative).as_deref(),
            )
            .is_none());

        let singleton_batch = InputIds::Batch(vec![vec![1, 2, 3]]);
        let singleton_tokens = PDRouter::p2p_tokens_from_input_ids(&singleton_batch).unwrap();
        assert_eq!(
            router
                .p2p_tokens_for_chat(&request, Some("glm"), Some(&singleton_tokens))
                .unwrap(),
            vec![1, 2, 3]
        );

        let batch = InputIds::Batch(vec![vec![1, 2], vec![3, 4]]);
        assert!(router
            .p2p_tokens_for_chat(
                &request,
                Some("glm"),
                PDRouter::p2p_tokens_from_input_ids(&batch).as_deref(),
            )
            .is_none());
    }

    #[tokio::test]
    async fn test_streaming_load_tracking() {
        use futures_util::StreamExt;
        use tokio::time::{sleep, Duration};

        let router = create_test_pd_router();

        let prefill_worker = create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(prefill_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let prefill_workers = router.worker_registry.get_prefill_workers();
        let decode_workers = router.worker_registry.get_decode_workers();

        let prefill_ref = prefill_workers[0].clone();
        let decode_ref = decode_workers[0].clone();

        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream = UnboundedReceiverStream::new(rx);

        {
            let response = router.create_streaming_response(
                stream.map(Ok),
                StatusCode::OK,
                None,
                false,
                None,
                decode_ref.clone(),
                vec![
                    WorkerLoadGuard::new(prefill_ref.clone(), None),
                    WorkerLoadGuard::new(decode_ref.clone(), None),
                ],
            );

            // Guards are now attached to response body, so load should be 1
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            tx.send(bytes::Bytes::from("test data")).unwrap();

            sleep(Duration::from_millis(10)).await;

            // Load still 1 while response body exists
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            drop(tx);

            // Response (and its body with guards) dropped here
            drop(response);
        }

        // Guards dropped when response dropped
        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);
    }
}

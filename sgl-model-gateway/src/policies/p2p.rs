use std::sync::Arc;

use super::kv_events::{
    compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle, HashTree,
};
use crate::core::Worker;

pub const DEFAULT_BOOTSTRAP_PORT: u16 = 8998;

#[derive(Debug, Clone, Copy)]
pub struct P2pRoutingConfig {
    pub cache_threshold: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteKvDecision {
    pub source_url: String,
    pub source_bootstrap_addr: String,
    pub target_url: String,
    pub matched_tokens: usize,
    pub token_ids: Vec<u32>,
    pub reason: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P2pSelection {
    pub target_index: usize,
    pub remote_kv: Option<RemoteKvDecision>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P2pSourceMatch {
    pub source_index: usize,
    pub matched_tokens: usize,
    pub source_bootstrap_addr: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P2pPreparedRequest {
    block_hashes: Vec<i64>,
    block_size: usize,
    boundary_tokens: usize,
}

/// Conservative Prefill-to-Prefill selector layered on top of the legacy
/// cache-aware policy. It only requests a transfer when a real KV-event hit
/// exists and load thresholds independently require moving away from that
/// cache owner. Missing routing metadata returns `None` so the caller can use
/// the legacy policy unchanged; a missing bootstrap port uses the SGLang
/// default.
#[derive(Debug)]
pub struct P2pCacheAwareSelector {
    config: P2pRoutingConfig,
    tree: Arc<HashTree>,
    block_size_oracle: Arc<BlockSizeOracle>,
}

impl P2pCacheAwareSelector {
    pub fn new(
        config: P2pRoutingConfig,
        tree: Arc<HashTree>,
        block_size_oracle: Arc<BlockSizeOracle>,
    ) -> Self {
        Self {
            config,
            tree,
            block_size_oracle,
        }
    }

    pub fn prepare_request(&self, token_ids: &[u32]) -> Option<P2pPreparedRequest> {
        if token_ids.is_empty() {
            return None;
        }
        let block_size = self.block_size_oracle.get()? as usize;
        let is_bigram = self.block_size_oracle.is_bigram();
        let block_hashes = if is_bigram {
            compute_block_hashes_bigram(token_ids, block_size)
        } else {
            compute_block_hashes(token_ids, block_size)
        };
        if block_hashes.is_empty() {
            return None;
        }
        Some(P2pPreparedRequest {
            block_hashes,
            block_size,
            boundary_tokens: usize::from(is_bigram),
        })
    }

    pub fn match_source_prepared(
        &self,
        workers: &[Arc<dyn Worker>],
        token_count: usize,
        prepared: &P2pPreparedRequest,
    ) -> Option<P2pSourceMatch> {
        if workers.is_empty() || token_count == 0 {
            return None;
        }

        let matched_by_worker = self
            .tree
            .match_prefix_by_worker(None, &prepared.block_hashes);
        let mut owner: Option<(usize, usize)> = None;
        for (index, worker) in workers.iter().enumerate() {
            let depth = matched_by_worker
                .iter()
                .filter(|(id, _)| id.url == worker.url())
                .map(|(_, depth)| *depth)
                .max()
                .unwrap_or(0);
            if depth == 0 {
                continue;
            }
            match owner {
                None => owner = Some((index, depth)),
                Some((best_index, best_depth))
                    if depth > best_depth
                        || (depth == best_depth
                            && workers[index].load() < workers[best_index].load()) =>
                {
                    owner = Some((index, depth));
                }
                _ => {}
            }
        }

        let (owner_index, matched_blocks) = owner?;
        let match_rate = matched_blocks as f32 / prepared.block_hashes.len() as f32;
        if match_rate <= self.config.cache_threshold {
            return None;
        }

        // EAGLE hashes overlapping token bigrams. N logical KV positions
        // therefore require N + 1 raw tokens to preserve the right-hand
        // boundary used by the worker.
        let matched_tokens = (matched_blocks * prepared.block_size)
            .saturating_add(prepared.boundary_tokens)
            .min(token_count);
        let source = &workers[owner_index];

        Some(P2pSourceMatch {
            source_index: owner_index,
            matched_tokens,
            source_bootstrap_addr: source_bootstrap_addr(
                source.url(),
                source.bootstrap_port().unwrap_or(DEFAULT_BOOTSTRAP_PORT),
            ),
        })
    }

    pub fn match_source(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: &[u32],
    ) -> Option<P2pSourceMatch> {
        let prepared = self.prepare_request(token_ids)?;
        self.match_source_prepared(workers, token_ids.len(), &prepared)
    }

    pub fn pair_is_beneficial(&self, source: &dyn Worker, target: &dyn Worker) -> bool {
        if !self.is_distinct_node(source, target) {
            return false;
        }

        let source_load = source.load();
        let target_load = target.load();
        let abs_diff = source_load.saturating_sub(target_load);
        abs_diff > self.config.balance_abs_threshold
            && (source_load as f32) > (target_load as f32 * self.config.balance_rel_threshold)
    }

    pub fn is_distinct_node(&self, source: &dyn Worker, target: &dyn Worker) -> bool {
        !same_node(source.url(), target.url())
    }

    fn selection_for_target_match(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: &[u32],
        source_match: P2pSourceMatch,
        target_index: usize,
    ) -> P2pSelection {
        let source = &workers[source_match.source_index];
        let target = &workers[target_index];
        let remote_kv = source_match
            .source_bootstrap_addr
            .filter(|_| self.pair_is_beneficial(source.as_ref(), target.as_ref()))
            .map(|source_bootstrap_addr| RemoteKvDecision {
                source_url: source.url().to_string(),
                source_bootstrap_addr,
                target_url: target.url().to_string(),
                matched_tokens: source_match.matched_tokens,
                token_ids: token_ids[..source_match.matched_tokens].to_vec(),
                reason: "load_imbalance",
            });

        P2pSelection {
            target_index: if remote_kv.is_some() {
                target_index
            } else {
                source_match.source_index
            },
            remote_kv,
        }
    }

    pub fn select_for_target(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: &[u32],
        target_url: &str,
    ) -> Option<P2pSelection> {
        let prepared = self.prepare_request(token_ids)?;
        self.select_for_target_prepared(workers, token_ids, &prepared, target_url)
    }

    pub fn select_for_target_prepared(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: &[u32],
        prepared: &P2pPreparedRequest,
        target_url: &str,
    ) -> Option<P2pSelection> {
        let source_match = self.match_source_prepared(workers, token_ids.len(), prepared)?;
        let target_index = workers
            .iter()
            .position(|worker| {
                worker.is_available()
                    && worker.url().trim_end_matches('/') == target_url.trim_end_matches('/')
            })
            .or_else(|| {
                workers
                    .iter()
                    .position(|worker| worker.is_available() && same_node(worker.url(), target_url))
            })?;
        Some(self.selection_for_target_match(workers, token_ids, source_match, target_index))
    }

    pub fn select(&self, workers: &[Arc<dyn Worker>], token_ids: &[u32]) -> Option<P2pSelection> {
        let prepared = self.prepare_request(token_ids)?;
        self.select_prepared(workers, token_ids, &prepared)
    }

    pub fn select_prepared(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: &[u32],
        prepared: &P2pPreparedRequest,
    ) -> Option<P2pSelection> {
        let source_match = self.match_source_prepared(workers, token_ids.len(), prepared)?;
        let min_index = workers
            .iter()
            .enumerate()
            .min_by_key(|(_, worker)| worker.load())
            .map(|(index, _)| index)?;
        Some(self.selection_for_target_match(workers, token_ids, source_match, min_index))
    }
}

fn same_node(left: &str, right: &str) -> bool {
    match (url::Url::parse(left), url::Url::parse(right)) {
        (Ok(left), Ok(right)) => {
            left.origin().ascii_serialization() == right.origin().ascii_serialization()
        }
        _ => left.trim_end_matches('/') == right.trim_end_matches('/'),
    }
}

fn source_bootstrap_addr(source_url: &str, port: u16) -> Option<String> {
    if port == 0 {
        return None;
    }
    let parsed = url::Url::parse(source_url).ok()?;
    let host = parsed.host_str()?;
    if host.contains(':') {
        Some(format!("[{host}]:{port}"))
    } else {
        Some(format!("{host}:{port}"))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::{
        core::{BasicWorkerBuilder, WorkerType},
        policies::kv_events::KvWorkerId,
    };

    fn worker(url: &str, bootstrap_port: Option<u16>, load: usize) -> Arc<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Prefill { bootstrap_port })
            .build();
        worker.load_counter.store(load, Ordering::Relaxed);
        Arc::new(worker)
    }

    fn selector(owner_url: &str) -> (P2pCacheAwareSelector, Vec<u32>) {
        let token_ids: Vec<u32> = (0..16).collect();
        let block_size = 4;
        let block_hashes = compute_block_hashes(&token_ids, block_size);
        let tree = Arc::new(HashTree::new());
        tree.insert(
            &KvWorkerId::new(owner_url.to_string(), 0),
            None,
            &block_hashes,
        );
        let oracle = BlockSizeOracle::new();
        oracle.try_set(block_size as u32).unwrap();
        (
            P2pCacheAwareSelector::new(
                P2pRoutingConfig {
                    cache_threshold: 0.5,
                    balance_abs_threshold: 2,
                    balance_rel_threshold: 1.2,
                },
                tree,
                oracle,
            ),
            token_ids,
        )
    }

    #[test]
    fn injects_remote_kv_when_first_worker_owns_cache_but_second_is_cold() {
        let source_url = "http://10.0.0.1:30000";
        let target_url = "http://10.0.0.2:30000";
        let (selector, token_ids) = selector(source_url);
        let workers = vec![
            worker(source_url, Some(32400), 10),
            worker(target_url, Some(32400), 1),
        ];

        let selection = selector.select(&workers, &token_ids).unwrap();

        assert_eq!(selection.target_index, 1);
        let remote = selection.remote_kv.unwrap();
        assert_eq!(remote.source_url, source_url);
        assert_eq!(remote.target_url, target_url);
        assert_eq!(remote.source_bootstrap_addr, "10.0.0.1:32400");
        assert_eq!(remote.matched_tokens, token_ids.len());
    }

    #[test]
    fn supports_the_reverse_transfer_direction_for_the_same_pair() {
        let target_url = "http://10.0.0.1:30000";
        let source_url = "http://10.0.0.2:30000";
        let (selector, token_ids) = selector(source_url);
        let workers = vec![
            worker(target_url, Some(32400), 1),
            worker(source_url, Some(32400), 10),
        ];

        let selection = selector.select(&workers, &token_ids).unwrap();

        assert_eq!(selection.target_index, 0);
        let remote = selection.remote_kv.unwrap();
        assert_eq!(remote.source_url, source_url);
        assert_eq!(remote.target_url, target_url);
    }

    #[test]
    fn missing_source_bootstrap_port_falls_back_to_default_port() {
        let source_url = "http://10.0.0.1:30000";
        let target_url = "http://10.0.0.2:30000";
        let (selector, token_ids) = selector(source_url);
        let workers = vec![
            worker(source_url, None, 10),
            worker(target_url, Some(32400), 1),
        ];

        let selection = selector.select(&workers, &token_ids).unwrap();

        assert_eq!(selection.target_index, 1);
        let remote = selection.remote_kv.unwrap();
        assert_eq!(remote.source_bootstrap_addr, "10.0.0.1:8998");
    }

    #[test]
    fn conservative_load_threshold_keeps_request_on_cache_owner() {
        let source_url = "http://10.0.0.1:30000";
        let target_url = "http://10.0.0.2:30000";
        let (selector, token_ids) = selector(source_url);
        let workers = vec![
            worker(source_url, Some(32400), 3),
            worker(target_url, Some(32400), 1),
        ];

        let selection = selector.select(&workers, &token_ids).unwrap();

        assert_eq!(selection.target_index, 0);
        assert!(selection.remote_kv.is_none());
    }

    #[test]
    fn unrelated_busy_worker_does_not_force_an_idle_owner_transfer() {
        let source_url = "http://10.0.0.1:30000";
        let target_url = "http://10.0.0.2:30000";
        let unrelated_url = "http://10.0.0.3:30000";
        let (selector, token_ids) = selector(source_url);
        let workers = vec![
            worker(source_url, Some(32400), 1),
            worker(target_url, Some(32400), 0),
            worker(unrelated_url, Some(32400), 100),
        ];

        let selection = selector.select(&workers, &token_ids).unwrap();

        assert_eq!(selection.target_index, 0);
        assert!(
            selection.remote_kv.is_none(),
            "an unrelated maximum load must not make an idle cache owner send KV"
        );
    }

    #[test]
    fn explicit_target_accepts_the_second_lowest_beneficial_worker() {
        let source_url = "http://10.0.0.1:30000";
        let lowest_url = "http://10.0.0.2:30000";
        let second_lowest_url = "http://10.0.0.3:30000";
        let (selector, token_ids) = selector(source_url);
        let workers = vec![
            worker(source_url, Some(32400), 10),
            worker(lowest_url, Some(32400), 1),
            worker(second_lowest_url, Some(32400), 2),
        ];

        let selection = selector
            .select_for_target(&workers, &token_ids, second_lowest_url)
            .unwrap();

        assert_eq!(selection.target_index, 2);
        let remote = selection.remote_kv.unwrap();
        assert_eq!(remote.source_url, source_url);
        assert_eq!(remote.target_url, second_lowest_url);
    }

    #[test]
    fn explicit_target_is_rejected_when_the_pair_gap_disappears() {
        let source_url = "http://10.0.0.1:30000";
        let target_url = "http://10.0.0.2:30000";
        let (selector, token_ids) = selector(source_url);
        let workers = vec![
            worker(source_url, Some(32400), 4),
            worker(target_url, Some(32400), 2),
        ];

        let selection = selector
            .select_for_target(&workers, &token_ids, target_url)
            .unwrap();

        assert_eq!(selection.target_index, 0);
        assert!(selection.remote_kv.is_none());
    }

    #[test]
    fn canonical_aliases_are_never_treated_as_a_p2p_pair() {
        let (selector, _) = selector("http://worker-a/");
        let source = worker("HTTP://WORKER-A:80/path", Some(32400), 10);
        let alias = worker("http://worker-a/", Some(32400), 1);

        assert!(
            !selector.pair_is_beneficial(source.as_ref(), alias.as_ref()),
            "policy and gate must use the same canonical node identity"
        );
    }

    #[test]
    fn bigram_match_preserves_the_raw_token_boundary() {
        let source_url = "http://10.0.0.1:30000";
        let target_url = "http://10.0.0.2:30000";
        let token_ids: Vec<u32> = (0..4096).collect();
        let block_size = 64;
        let block_hashes = compute_block_hashes_bigram(&token_ids, block_size);
        assert_eq!(block_hashes.len(), 64);

        let tree = Arc::new(HashTree::new());
        tree.insert(
            &KvWorkerId::new(source_url.to_string(), 0),
            None,
            &block_hashes[..63],
        );
        let oracle = BlockSizeOracle::new();
        oracle.try_set(block_size as u32).unwrap();
        oracle.set_bigram(true);
        let selector = P2pCacheAwareSelector::new(
            P2pRoutingConfig {
                cache_threshold: 0.5,
                balance_abs_threshold: 2,
                balance_rel_threshold: 1.2,
            },
            tree,
            oracle,
        );
        let workers = vec![
            worker(source_url, Some(32400), 10),
            worker(target_url, Some(32400), 1),
        ];

        let selection = selector.select(&workers, &token_ids).unwrap();
        let remote = selection.remote_kv.unwrap();

        assert_eq!(remote.matched_tokens, 4033);
        assert_eq!(remote.token_ids.len(), 4033);
    }
}

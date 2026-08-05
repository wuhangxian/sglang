use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, OnceLock},
    time::Duration,
};

use parking_lot::Mutex;
use tokio::{
    sync::Notify,
    time::{sleep_until, Instant},
};

use crate::core::Worker;

static PROCESS_P2P_NODE_COORDINATOR: OnceLock<Arc<P2pNodeCoordinator>> = OnceLock::new();

#[derive(Clone)]
struct P2pNodeCandidate {
    node_key: String,
    worker: Option<Arc<dyn Worker>>,
}

impl P2pNodeCandidate {
    fn worker(worker: Arc<dyn Worker>) -> Self {
        Self {
            node_key: canonical_node_key(worker.url()),
            worker: Some(worker),
        }
    }

    fn fixed(endpoint: &str) -> Self {
        Self {
            node_key: canonical_node_key(endpoint),
            worker: None,
        }
    }

    fn is_available(&self) -> bool {
        self.worker
            .as_ref()
            .is_none_or(|worker| worker.is_available())
    }

    fn load(&self) -> usize {
        self.worker.as_ref().map_or(0, |worker| worker.load())
    }
}

struct P2pWaiterState {
    waiter_id: u64,
    source: Option<P2pNodeCandidate>,
    candidates: Vec<P2pNodeCandidate>,
    suspended: bool,
    protected_target: Option<String>,
}

#[derive(Default)]
struct P2pNodeState {
    owners: HashMap<String, u64>,
    next_owner_id: u64,
    waiters: VecDeque<P2pWaiterState>,
    next_waiter_id: u64,
}

#[derive(Default)]
struct P2pNodeCoordinator {
    state: Mutex<P2pNodeState>,
    changed: Notify,
}

impl P2pNodeCoordinator {
    fn register_waiter(self: &Arc<Self>) -> P2pWaiterRegistration {
        let waiter_id = {
            let mut state = self.state.lock();
            state.next_waiter_id = state.next_waiter_id.wrapping_add(1);
            let waiter_id = state.next_waiter_id;
            state.waiters.push_back(P2pWaiterState {
                waiter_id,
                source: None,
                candidates: Vec::new(),
                suspended: false,
                protected_target: None,
            });
            waiter_id
        };
        self.changed.notify_waiters();
        P2pWaiterRegistration {
            coordinator: Arc::clone(self),
            waiter_id,
            registered: true,
        }
    }

    fn remove_waiter(&self, waiter_id: u64) {
        let mut state = self.state.lock();
        let original_len = state.waiters.len();
        state.waiters.retain(|waiter| waiter.waiter_id != waiter_id);
        let removed = state.waiters.len() != original_len;
        drop(state);
        if removed {
            self.changed.notify_waiters();
        }
    }

    fn suspend_waiter(&self, waiter_id: u64, protected_target: String) {
        let mut state = self.state.lock();
        let suspended = state
            .waiters
            .iter_mut()
            .find(|waiter| waiter.waiter_id == waiter_id)
            .map(|waiter| {
                waiter.suspended = true;
                waiter.protected_target = Some(protected_target);
            })
            .is_some();
        drop(state);
        if suspended {
            // Preserve the ticket position, but do not let a rejected stale
            // plan block otherwise satisfiable (including disjoint) waiters
            // while this task waits for its bounded replan tick.
            self.changed.notify_waiters();
        }
    }

    fn try_acquire_best(
        self: &Arc<Self>,
        waiter_id: u64,
        source: P2pNodeCandidate,
        candidates: Vec<P2pNodeCandidate>,
    ) -> Option<(P2pNodeCandidate, P2pNodeLease)> {
        let mut state = self.state.lock();
        let mut candidates_by_key = HashMap::<String, P2pNodeCandidate>::new();
        for candidate in candidates
            .into_iter()
            .filter(|candidate| candidate.node_key != source.node_key)
        {
            match candidates_by_key.entry(candidate.node_key.clone()) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(candidate);
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let current = entry.get();
                    if (!current.is_available() && candidate.is_available())
                        || (current.is_available()
                            && candidate.is_available()
                            && candidate.load() < current.load())
                    {
                        entry.insert(candidate);
                    }
                }
            }
        }
        let mut candidates = candidates_by_key.into_values().collect::<Vec<_>>();
        candidates.sort_by(|left, right| left.node_key.cmp(&right.node_key));

        let waiter = state
            .waiters
            .iter_mut()
            .find(|waiter| waiter.waiter_id == waiter_id)?;
        waiter.source = Some(source);
        waiter.candidates = candidates;
        waiter.suspended = false;
        waiter.protected_target = None;

        let protected_by_earlier_waiters = |waiter_index: usize| -> HashSet<&str> {
            let mut protected = HashSet::new();
            for waiter in state.waiters.iter().take(waiter_index) {
                // A source is mandatory for its waiter, so a younger request
                // must never keep an older request unsatisfied by borrowing
                // that free source for a different target.
                if let Some(source) = waiter.source.as_ref() {
                    protected.insert(source.node_key.as_str());
                }
                // A suspended waiter cannot react to release notifications
                // until its bounded replan tick. Protect only the target it
                // just provisionally selected, rather than its full candidate
                // set, so unrelated pairs remain concurrent.
                if waiter.suspended {
                    if let Some(target) = waiter.protected_target.as_deref() {
                        protected.insert(target);
                    }
                }
            }
            protected
        };
        let winner = state
            .waiters
            .iter()
            .enumerate()
            .find_map(|(waiter_index, waiter)| {
                if waiter.suspended {
                    return None;
                }
                let source = waiter.source.as_ref()?;
                let protected = protected_by_earlier_waiters(waiter_index);
                if !source.is_available()
                    || state.owners.contains_key(&source.node_key)
                    || protected.contains(source.node_key.as_str())
                {
                    return None;
                }
                waiter
                    .candidates
                    .iter()
                    .any(|candidate| {
                        candidate.is_available()
                            && !state.owners.contains_key(&candidate.node_key)
                            && !protected.contains(candidate.node_key.as_str())
                    })
                    .then_some(waiter.waiter_id)
            });
        if winner != Some(waiter_id) {
            return None;
        }

        let waiter_index = state
            .waiters
            .iter()
            .position(|waiter| waiter.waiter_id == waiter_id)?;
        let protected = protected_by_earlier_waiters(waiter_index);
        let selected = state.waiters[waiter_index]
            .candidates
            .iter()
            .filter(|candidate| {
                candidate.is_available()
                    && !state.owners.contains_key(&candidate.node_key)
                    && !protected.contains(candidate.node_key.as_str())
            })
            .min_by(|left, right| {
                left.load()
                    .cmp(&right.load())
                    .then_with(|| left.node_key.cmp(&right.node_key))
            })?
            .clone();
        let source = state.waiters[waiter_index].source.as_ref()?.clone();

        state.next_owner_id = state.next_owner_id.wrapping_add(1);
        let owner_id = state.next_owner_id;
        let mut node_keys = vec![source.node_key, selected.node_key.clone()];
        node_keys.sort_unstable();
        node_keys.dedup();
        if node_keys.len() != 2 || node_keys.iter().any(|key| state.owners.contains_key(key)) {
            return None;
        }
        for key in &node_keys {
            state.owners.insert(key.clone(), owner_id);
        }
        drop(state);
        self.changed.notify_waiters();

        Some((
            selected,
            P2pNodeLease {
                coordinator: Arc::clone(self),
                node_keys,
                owner_id,
            },
        ))
    }

    fn release(&self, node_keys: &[String], owner_id: u64) {
        let mut state = self.state.lock();
        for key in node_keys {
            if state.owners.get(key) == Some(&owner_id) {
                state.owners.remove(key);
            }
        }
        drop(state);
        self.changed.notify_waiters();
    }
}

struct P2pWaiterRegistration {
    coordinator: Arc<P2pNodeCoordinator>,
    waiter_id: u64,
    registered: bool,
}

impl P2pWaiterRegistration {
    fn remove(&mut self) {
        if self.registered {
            self.coordinator.remove_waiter(self.waiter_id);
            self.registered = false;
        }
    }
}

impl Drop for P2pWaiterRegistration {
    fn drop(&mut self) {
        self.remove();
    }
}

pub(super) enum P2pFreshPlan<T> {
    Candidate {
        source: Arc<dyn Worker>,
        candidates: Vec<Arc<dyn Worker>>,
        context: T,
    },
    Stop {
        reason: &'static str,
        fallback: Option<Arc<dyn Worker>>,
    },
}

pub(super) enum P2pAdmissionResult<T> {
    Granted {
        context: T,
        target: Arc<dyn Worker>,
        lease: P2pNodeLease,
    },
    Stopped {
        reason: &'static str,
        fallback: Option<Arc<dyn Worker>>,
    },
    TimedOut,
}

#[derive(Clone)]
pub(super) struct P2pNodeGate {
    coordinator: Arc<P2pNodeCoordinator>,
    retry_interval: Duration,
    max_retries: usize,
}

impl P2pNodeGate {
    pub(super) fn new(retry_interval: Duration, max_retries: usize) -> Self {
        Self {
            coordinator: Arc::clone(
                PROCESS_P2P_NODE_COORDINATOR
                    .get_or_init(|| Arc::new(P2pNodeCoordinator::default())),
            ),
            retry_interval,
            max_retries,
        }
    }

    #[cfg(test)]
    pub(super) fn new_isolated(retry_interval: Duration, max_retries: usize) -> Self {
        Self {
            coordinator: Arc::new(P2pNodeCoordinator::default()),
            retry_interval,
            max_retries,
        }
    }

    /// Atomically reserve both P2P endpoints.
    ///
    /// This fixed-pair wrapper is retained for focused lock tests. Production
    /// admission uses `acquire_best_with`, which keeps a cancellation-safe
    /// ticket and atomically selects the lowest-load unlocked target.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) async fn acquire(&self, source_url: &str, target_url: &str) -> Option<P2pNodeLease> {
        let deadline = Instant::now() + self.retry_interval * self.max_retries as u32;
        let mut registration = self.coordinator.register_waiter();
        let source = P2pNodeCandidate::fixed(source_url);
        let target = P2pNodeCandidate::fixed(target_url);
        let mut next_retry = Instant::now() + self.retry_interval;

        loop {
            let changed = self.coordinator.changed.notified();
            if let Some((_, lease)) = self.coordinator.try_acquire_best(
                registration.waiter_id,
                source.clone(),
                vec![target.clone()],
            ) {
                registration.remove();
                return Some(lease);
            }
            if Instant::now() >= deadline {
                return None;
            }
            tokio::select! {
                biased;
                () = sleep_until(deadline) => return None,
                () = sleep_until(next_retry) => {
                    next_retry = (Instant::now() + self.retry_interval).min(deadline);
                }
                () = changed => {}
            }
        }
    }

    pub(super) async fn acquire_best_with<T, U, F, V>(
        &self,
        deadline: Instant,
        mut fresh_plan: F,
        mut validate: V,
    ) -> P2pAdmissionResult<U>
    where
        F: FnMut() -> P2pFreshPlan<T>,
        V: FnMut(&T, &Arc<dyn Worker>) -> Option<U>,
    {
        let mut registration = self.coordinator.register_waiter();
        let mut next_replan = (Instant::now() + self.retry_interval).min(deadline);
        let mut cached_plan = None;

        loop {
            if Instant::now() >= deadline {
                return P2pAdmissionResult::TimedOut;
            }

            if cached_plan.is_none() {
                cached_plan = match fresh_plan() {
                    P2pFreshPlan::Candidate {
                        source,
                        candidates,
                        context,
                    } => Some((source, candidates, context)),
                    P2pFreshPlan::Stop { reason, fallback } => {
                        registration.remove();
                        return P2pAdmissionResult::Stopped { reason, fallback };
                    }
                };
                if Instant::now() >= deadline {
                    return P2pAdmissionResult::TimedOut;
                }
            }
            let (source, candidates, context) =
                cached_plan.as_ref().expect("fresh plan must be cached");

            // Register for changes before trying the atomic selection so a
            // release between the failed check and await cannot be missed.
            let changed = self.coordinator.changed.notified();
            if let Some((selected, lease)) = self.coordinator.try_acquire_best(
                registration.waiter_id,
                P2pNodeCandidate::worker(Arc::clone(source)),
                candidates
                    .iter()
                    .cloned()
                    .map(P2pNodeCandidate::worker)
                    .collect(),
            ) {
                let protected_target = selected.node_key.clone();
                let target = selected
                    .worker
                    .expect("dynamic P2P candidate must retain its worker");
                if Instant::now() >= deadline {
                    drop(lease);
                    return P2pAdmissionResult::TimedOut;
                }
                let validated = validate(&context, &target);
                if Instant::now() >= deadline {
                    drop(lease);
                    return P2pAdmissionResult::TimedOut;
                }
                if let Some(validated) = validated {
                    registration.remove();
                    return P2pAdmissionResult::Granted {
                        context: validated,
                        target,
                        lease,
                    };
                }
                // The pair changed between the fresh plan and final
                // validation. Keep the same waiter ticket, release only the
                // provisional node lease, and replan on the bounded tick. Do
                // not consume the release notification produced by our own
                // provisional lease: a stable rejection must not busy-loop.
                drop(lease);
                self.coordinator
                    .suspend_waiter(registration.waiter_id, protected_target);
                cached_plan = None;
                tokio::select! {
                    biased;
                    () = sleep_until(deadline) => return P2pAdmissionResult::TimedOut,
                    () = sleep_until(next_replan) => {
                        next_replan =
                            (Instant::now() + self.retry_interval).min(deadline);
                    }
                }
                continue;
            }

            tokio::select! {
                biased;
                () = sleep_until(deadline) => return P2pAdmissionResult::TimedOut,
                () = sleep_until(next_replan) => {
                    next_replan = (Instant::now() + self.retry_interval).min(deadline);
                    cached_plan = None;
                }
                () = changed => {}
            }
        }
    }
}

pub(super) struct P2pNodeLease {
    coordinator: Arc<P2pNodeCoordinator>,
    node_keys: Vec<String>,
    owner_id: u64,
}

impl Drop for P2pNodeLease {
    fn drop(&mut self) {
        self.coordinator.release(&self.node_keys, self.owner_id);
    }
}

fn canonical_node_key(endpoint: &str) -> String {
    url::Url::parse(endpoint)
        .ok()
        .map(|url| url.origin().ascii_serialization())
        .unwrap_or_else(|| endpoint.trim_end_matches('/').to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tokio::time::{sleep, timeout};

    fn worker(url: &str, load: usize) -> Arc<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(32400),
            })
            .build();
        worker.load_counter.store(load, Ordering::Relaxed);
        Arc::new(worker)
    }

    async fn acquire_best(
        gate: &P2pNodeGate,
        source: Arc<dyn Worker>,
        candidates: Vec<Arc<dyn Worker>>,
    ) -> P2pAdmissionResult<()> {
        gate.acquire_best_with(
            Instant::now() + Duration::from_secs(1),
            || P2pFreshPlan::Candidate {
                source: Arc::clone(&source),
                candidates: candidates.clone(),
                context: (),
            },
            |_, _target| Some(()),
        )
        .await
    }

    #[tokio::test]
    async fn stopped_plan_returns_its_fresh_local_fallback() {
        let gate = P2pNodeGate::new_isolated(Duration::from_secs(1), 1);
        let fallback = worker("http://worker-a:30000", 10);
        let result = gate
            .acquire_best_with(
                Instant::now() + Duration::from_secs(1),
                || P2pFreshPlan::<()>::Stop {
                    reason: "no_transfer_needed",
                    fallback: Some(Arc::clone(&fallback)),
                },
                |_, _| Some(()),
            )
            .await;

        match result {
            P2pAdmissionResult::Stopped {
                reason,
                fallback: Some(worker),
            } => {
                assert_eq!(reason, "no_transfer_needed");
                assert_eq!(worker.url(), fallback.url());
            }
            _ => panic!("stopped plan must preserve its fresh local owner"),
        }
    }

    #[tokio::test]
    async fn disjoint_pairs_can_run_concurrently() {
        let gate = P2pNodeGate::new_isolated(Duration::from_secs(1), 1);
        let first = gate
            .acquire("http://worker-a:30000", "http://worker-b:30000")
            .await
            .expect("first pair must enter");

        let second = timeout(
            Duration::from_millis(20),
            gate.acquire("http://worker-c:30000", "http://worker-d:30000"),
        )
        .await
        .expect("disjoint pair must not wait")
        .expect("disjoint pair must enter");

        drop((first, second));
    }

    #[tokio::test]
    async fn fair_admission_does_not_serialize_disjoint_transfers() {
        let gate = P2pNodeGate::new_isolated(Duration::from_millis(20), 50);
        let first = acquire_best(
            &gate,
            worker("http://worker-a:30000", 10),
            vec![worker("http://worker-b:30000", 1)],
        )
        .await;
        let first_lease = match first {
            P2pAdmissionResult::Granted { lease, .. } => lease,
            _ => panic!("first disjoint pair must enter"),
        };

        let second = timeout(
            Duration::from_millis(50),
            acquire_best(
                &gate,
                worker("http://worker-c:30000", 10),
                vec![worker("http://worker-d:30000", 1)],
            ),
        )
        .await
        .expect("fair queue must immediately admit a disjoint pair");
        let second_lease = match second {
            P2pAdmissionResult::Granted { lease, .. } => lease,
            _ => panic!("second disjoint pair must enter"),
        };

        drop((first_lease, second_lease));
    }

    #[tokio::test]
    async fn locked_lowest_target_falls_through_to_next_lowest_target() {
        let gate = P2pNodeGate::new_isolated(Duration::from_millis(20), 50);
        let lowest = worker("http://worker-b:30000", 1);
        let next_lowest = worker("http://worker-c:30000", 2);
        let blocker = gate
            .acquire(lowest.url(), "http://worker-x:30000")
            .await
            .expect("lowest target must be locked for the test");

        let result = acquire_best(
            &gate,
            worker("http://worker-a:30000", 10),
            vec![Arc::clone(&lowest), Arc::clone(&next_lowest)],
        )
        .await;
        let (target, lease) = match result {
            P2pAdmissionResult::Granted { target, lease, .. } => (target, lease),
            _ => panic!("the next-lowest unlocked target must be admitted"),
        };
        assert_eq!(target.url(), next_lowest.url());

        drop((lease, blocker));
    }

    #[tokio::test]
    async fn unsatisfiable_head_does_not_block_a_disjoint_waiter() {
        let gate = P2pNodeGate::new_isolated(Duration::from_millis(20), 50);
        let active = gate
            .acquire("http://worker-a:30000", "http://worker-x:30000")
            .await
            .expect("head source must start locked");
        let blocked = tokio::spawn({
            let gate = gate.clone();
            async move {
                acquire_best(
                    &gate,
                    worker("http://worker-a:30000", 10),
                    vec![worker("http://worker-b:30000", 1)],
                )
                .await
            }
        });
        sleep(Duration::from_millis(20)).await;

        let disjoint = timeout(
            Duration::from_millis(50),
            acquire_best(
                &gate,
                worker("http://worker-c:30000", 10),
                vec![worker("http://worker-d:30000", 1)],
            ),
        )
        .await
        .expect("the oldest satisfiable waiter must bypass an impossible head");
        let disjoint_lease = match disjoint {
            P2pAdmissionResult::Granted { lease, .. } => lease,
            _ => panic!("disjoint waiter must be granted"),
        };

        blocked.abort();
        assert!(blocked.await.is_err());
        drop((disjoint_lease, active));
    }

    #[tokio::test]
    async fn younger_waiter_cannot_borrow_an_older_waiters_free_source() {
        let gate = P2pNodeGate::new_isolated(Duration::from_millis(20), 50);
        let blocked_target = gate
            .acquire("http://worker-b:30000", "http://worker-x:30000")
            .await
            .expect("older waiter's target must start locked");
        let older = tokio::spawn({
            let gate = gate.clone();
            async move {
                acquire_best(
                    &gate,
                    worker("http://worker-a:30000", 10),
                    vec![worker("http://worker-b:30000", 1)],
                )
                .await
            }
        });
        sleep(Duration::from_millis(20)).await;

        let younger = tokio::spawn({
            let gate = gate.clone();
            async move {
                acquire_best(
                    &gate,
                    worker("http://worker-a:30000", 10),
                    vec![worker("http://worker-c:30000", 1)],
                )
                .await
            }
        });
        sleep(Duration::from_millis(30)).await;
        assert!(
            !younger.is_finished(),
            "a younger waiter must not take a free source already claimed by an older ticket"
        );

        drop(blocked_target);
        let older_result = timeout(Duration::from_millis(100), older)
            .await
            .expect("older waiter must run when its target releases")
            .unwrap();
        let older_lease = match older_result {
            P2pAdmissionResult::Granted { lease, .. } => lease,
            _ => panic!("older waiter must be granted first"),
        };
        assert!(
            !younger.is_finished(),
            "younger waiter must remain behind the older source lease"
        );

        drop(older_lease);
        let younger_result = timeout(Duration::from_millis(100), younger)
            .await
            .expect("younger waiter must run after the older lease releases")
            .unwrap();
        assert!(matches!(younger_result, P2pAdmissionResult::Granted { .. }));
    }

    #[tokio::test]
    async fn older_waiter_gets_target_after_one_already_running_conflict_finishes() {
        let gate = P2pNodeGate::new_isolated(Duration::from_millis(20), 50);
        let source_blocker = gate
            .acquire("http://worker-a:30000", "http://worker-x:30000")
            .await
            .expect("older waiter's source must start locked");
        let older = tokio::spawn({
            let gate = gate.clone();
            async move {
                acquire_best(
                    &gate,
                    worker("http://worker-a:30000", 10),
                    vec![worker("http://worker-b:30000", 1)],
                )
                .await
            }
        });
        sleep(Duration::from_millis(20)).await;

        let running_conflict = acquire_best(
            &gate,
            worker("http://worker-c:30000", 10),
            vec![worker("http://worker-b:30000", 1)],
        )
        .await;
        let running_lease = match running_conflict {
            P2pAdmissionResult::Granted { lease, .. } => lease,
            _ => panic!("a target conflict may run while the older source is unavailable"),
        };
        drop(source_blocker);
        assert!(
            !older.is_finished(),
            "older waiter must still wait for the already-running target owner"
        );

        let later = tokio::spawn({
            let gate = gate.clone();
            async move {
                acquire_best(
                    &gate,
                    worker("http://worker-d:30000", 10),
                    vec![worker("http://worker-b:30000", 1)],
                )
                .await
            }
        });
        drop(running_lease);

        let older_result = timeout(Duration::from_millis(100), older)
            .await
            .expect("older waiter must win the target on release")
            .unwrap();
        let older_lease = match older_result {
            P2pAdmissionResult::Granted { lease, .. } => lease,
            _ => panic!("older waiter must be granted before later target conflicts"),
        };
        assert!(
            !later.is_finished(),
            "a later target conflict must not steal the released target"
        );

        drop(older_lease);
        let later_result = timeout(Duration::from_millis(100), later)
            .await
            .expect("later waiter must run after the older lease releases")
            .unwrap();
        assert!(matches!(later_result, P2pAdmissionResult::Granted { .. }));
    }

    #[tokio::test]
    async fn failed_final_validation_keeps_the_original_waiter_age() {
        let gate = P2pNodeGate::new_isolated(Duration::from_millis(20), 50);
        let active = gate
            .acquire("http://worker-a:30000", "http://worker-x:30000")
            .await
            .expect("shared source must start locked");
        let validation_calls = Arc::new(AtomicUsize::new(0));
        let first = tokio::spawn({
            let gate = gate.clone();
            let validation_calls = Arc::clone(&validation_calls);
            async move {
                let source = worker("http://worker-a:30000", 10);
                let target = worker("http://worker-b:30000", 1);
                gate.acquire_best_with(
                    Instant::now() + Duration::from_secs(1),
                    || P2pFreshPlan::Candidate {
                        source: Arc::clone(&source),
                        candidates: vec![Arc::clone(&target)],
                        context: (),
                    },
                    |_, _target| {
                        (validation_calls.fetch_add(1, Ordering::AcqRel) > 0).then_some(())
                    },
                )
                .await
            }
        });
        sleep(Duration::from_millis(20)).await;
        let second = tokio::spawn({
            let gate = gate.clone();
            async move {
                acquire_best(
                    &gate,
                    worker("http://worker-a:30000", 10),
                    vec![worker("http://worker-c:30000", 1)],
                )
                .await
            }
        });

        drop(active);
        let first_result = timeout(Duration::from_millis(200), first)
            .await
            .expect("oldest waiter must finish")
            .unwrap();
        let first_lease = match first_result {
            P2pAdmissionResult::Granted { lease, .. } => lease,
            _ => panic!("oldest waiter must retain priority after stale validation"),
        };
        assert!(
            !second.is_finished(),
            "a younger waiter must not steal the shared source"
        );

        drop(first_lease);
        let second_result = timeout(Duration::from_millis(200), second)
            .await
            .expect("younger waiter must run after the old lease releases")
            .unwrap();
        assert!(matches!(second_result, P2pAdmissionResult::Granted { .. }));
    }

    #[tokio::test]
    async fn stable_validation_rejection_is_tick_bounded_not_a_busy_loop() {
        let interval = Duration::from_millis(20);
        let gate = P2pNodeGate::new_isolated(interval, 10);
        let validation_calls = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&validation_calls);
        let source = worker("http://worker-a:30000", 10);
        let target = worker("http://worker-b:30000", 1);

        let result = gate
            .acquire_best_with(
                Instant::now() + Duration::from_millis(75),
                || P2pFreshPlan::Candidate {
                    source: Arc::clone(&source),
                    candidates: vec![Arc::clone(&target)],
                    context: (),
                },
                |_, _target| {
                    calls.fetch_add(1, Ordering::AcqRel);
                    None::<()>
                },
            )
            .await;

        assert!(matches!(result, P2pAdmissionResult::TimedOut));
        let calls = validation_calls.load(Ordering::Acquire);
        assert!(
            (2..=6).contains(&calls),
            "stable rejection must retry on ticks, not spin: calls={calls}"
        );
        assert!(
            gate.acquire(source.url(), target.url()).await.is_some(),
            "timed-out validation must release both provisional nodes"
        );
    }

    #[tokio::test]
    async fn final_validation_must_finish_before_the_absolute_deadline() {
        let gate = P2pNodeGate::new_isolated(Duration::from_secs(1), 1);
        let source = worker("http://worker-a:30000", 10);
        let target = worker("http://worker-b:30000", 1);
        let result = gate
            .acquire_best_with(
                Instant::now() + Duration::from_millis(10),
                || P2pFreshPlan::Candidate {
                    source: Arc::clone(&source),
                    candidates: vec![Arc::clone(&target)],
                    context: (),
                },
                |_, _target| {
                    std::thread::sleep(Duration::from_millis(20));
                    Some(())
                },
            )
            .await;

        assert!(matches!(result, P2pAdmissionResult::TimedOut));
        assert!(
            gate.acquire(source.url(), target.url()).await.is_some(),
            "a validator that crosses the deadline must release both nodes"
        );
    }

    #[tokio::test]
    async fn suspended_stale_intent_allows_disjoint_bypass_but_protects_conflicts() {
        let interval = Duration::from_millis(200);
        let gate = P2pNodeGate::new_isolated(interval, 10);
        let validation_calls = Arc::new(AtomicUsize::new(0));
        let stale = tokio::spawn({
            let gate = gate.clone();
            let validation_calls = Arc::clone(&validation_calls);
            async move {
                let source = worker("http://worker-a:30000", 10);
                let selected_target = worker("http://worker-b:30000", 1);
                let unselected_candidate = worker("http://worker-c:30000", 2);
                gate.acquire_best_with(
                    Instant::now() + Duration::from_secs(2),
                    || P2pFreshPlan::Candidate {
                        source: Arc::clone(&source),
                        candidates: vec![
                            Arc::clone(&selected_target),
                            Arc::clone(&unselected_candidate),
                        ],
                        context: (),
                    },
                    |_, _target| {
                        validation_calls.fetch_add(1, Ordering::AcqRel);
                        None::<()>
                    },
                )
                .await
            }
        });
        timeout(Duration::from_millis(100), async {
            while validation_calls.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("stale waiter must reach final validation");

        let disjoint = timeout(
            Duration::from_millis(50),
            acquire_best(
                &gate,
                worker("http://worker-c:30000", 10),
                vec![worker("http://worker-d:30000", 1)],
            ),
        )
        .await
        .expect("disjoint pair must bypass a suspended stale intent");
        let disjoint_lease = match disjoint {
            P2pAdmissionResult::Granted { lease, .. } => lease,
            _ => panic!("disjoint pair must be granted"),
        };

        let conflicting = tokio::spawn({
            let gate = gate.clone();
            async move {
                acquire_best(
                    &gate,
                    worker("http://worker-a:30000", 10),
                    vec![worker("http://worker-e:30000", 1)],
                )
                .await
            }
        });
        let target_conflicting = tokio::spawn({
            let gate = gate.clone();
            async move {
                acquire_best(
                    &gate,
                    worker("http://worker-f:30000", 10),
                    vec![worker("http://worker-b:30000", 1)],
                )
                .await
            }
        });
        sleep(Duration::from_millis(30)).await;
        assert!(
            !conflicting.is_finished(),
            "younger use of a stale waiter's source must remain protected"
        );
        assert!(
            !target_conflicting.is_finished(),
            "the provisionally selected stale target must remain protected"
        );

        stale.abort();
        assert!(stale.await.is_err());
        let conflicting_result = timeout(Duration::from_millis(100), conflicting)
            .await
            .expect("cancelling stale intent must wake conflict")
            .unwrap();
        assert!(matches!(
            conflicting_result,
            P2pAdmissionResult::Granted { .. }
        ));
        let target_conflicting_result = timeout(Duration::from_millis(100), target_conflicting)
            .await
            .expect("cancelling stale intent must wake target conflict")
            .unwrap();
        assert!(matches!(
            target_conflicting_result,
            P2pAdmissionResult::Granted { .. }
        ));
        drop(disjoint_lease);
    }

    #[tokio::test]
    async fn release_wakeup_selects_using_current_candidate_loads() {
        let gate = P2pNodeGate::new_isolated(Duration::from_secs(1), 2);
        let source = worker("http://worker-a:30000", 10);
        let originally_lowest = worker("http://worker-b:30000", 1);
        let newly_lowest = worker("http://worker-c:30000", 2);
        let active = gate
            .acquire(source.url(), "http://worker-x:30000")
            .await
            .expect("source must start locked");
        let waiting = tokio::spawn({
            let gate = gate.clone();
            let source = Arc::clone(&source);
            let originally_lowest = Arc::clone(&originally_lowest);
            let newly_lowest = Arc::clone(&newly_lowest);
            async move { acquire_best(&gate, source, vec![originally_lowest, newly_lowest]).await }
        });
        sleep(Duration::from_millis(20)).await;
        // The test helper returns BasicWorker behind a trait object; update
        // loads through the Worker API so the gate observes them under its
        // coordinator mutex.
        for _ in 0..5 {
            originally_lowest.increment_load();
        }
        newly_lowest.decrement_load();
        newly_lowest.decrement_load();
        drop(active);

        let result = timeout(Duration::from_millis(100), waiting)
            .await
            .expect("release must wake the waiting plan")
            .unwrap();
        let target = match result {
            P2pAdmissionResult::Granted { target, .. } => target,
            _ => panic!("waiting plan must be granted"),
        };
        assert_eq!(target.url(), newly_lowest.url());
    }

    #[tokio::test]
    async fn sharing_either_endpoint_serializes_pairs() {
        let gate = P2pNodeGate::new_isolated(Duration::from_secs(1), 1);
        let active = gate
            .acquire("http://worker-a:30000", "http://worker-b:30000")
            .await
            .expect("active pair must enter");

        let source_conflict = tokio::spawn({
            let gate = gate.clone();
            async move {
                gate.acquire("http://worker-a:30000", "http://worker-c:30000")
                    .await
            }
        });
        let target_conflict = tokio::spawn({
            let gate = gate.clone();
            async move {
                gate.acquire("http://worker-d:30000", "http://worker-b:30000")
                    .await
            }
        });

        sleep(Duration::from_millis(20)).await;
        assert!(!source_conflict.is_finished());
        assert!(!target_conflict.is_finished());

        drop(active);
        assert!(timeout(Duration::from_millis(100), source_conflict)
            .await
            .expect("source-conflicting pair must wake")
            .unwrap()
            .is_some());
        assert!(timeout(Duration::from_millis(100), target_conflict)
            .await
            .expect("target-conflicting pair must wake")
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn reverse_direction_uses_the_same_two_node_lock() {
        let gate = P2pNodeGate::new_isolated(Duration::from_secs(1), 1);
        let forward = gate
            .acquire("http://worker-a:30000", "http://worker-b:30000")
            .await
            .expect("forward pair must enter");
        let reverse = tokio::spawn({
            let gate = gate.clone();
            async move {
                gate.acquire("http://worker-b:30000", "http://worker-a:30000")
                    .await
            }
        });

        sleep(Duration::from_millis(20)).await;
        assert!(!reverse.is_finished());
        drop(forward);
        assert!(timeout(Duration::from_millis(100), reverse)
            .await
            .expect("reverse pair must wake without deadlocking")
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn blocked_pair_does_not_block_disjoint_nodes() {
        let gate = P2pNodeGate::new_isolated(Duration::from_secs(1), 1);
        let active = gate
            .acquire("http://worker-b:30000", "http://worker-c:30000")
            .await
            .expect("active pair must enter");
        let blocked = tokio::spawn({
            let gate = gate.clone();
            async move {
                gate.acquire("http://worker-a:30000", "http://worker-b:30000")
                    .await
            }
        });

        sleep(Duration::from_millis(20)).await;
        let disjoint = timeout(
            Duration::from_millis(20),
            gate.acquire("http://worker-d:30000", "http://worker-e:30000"),
        )
        .await
        .expect("blocked A-B must not serialize disjoint D-E")
        .expect("D-E must enter");

        blocked.abort();
        drop((active, disjoint));
    }

    #[tokio::test]
    async fn three_retry_intervals_exhaust_the_budget_without_leaking_nodes() {
        let interval = Duration::from_millis(20);
        let gate = P2pNodeGate::new_isolated(interval, 3);
        let active = gate
            .acquire("http://worker-a:30000", "http://worker-b:30000")
            .await
            .expect("active pair must enter");
        let started = Instant::now();

        assert!(gate
            .acquire("http://worker-a:30000", "http://worker-c:30000")
            .await
            .is_none());
        assert!(started.elapsed() >= interval * 3);

        drop(active);
        assert!(
            gate.acquire("http://worker-a:30000", "http://worker-c:30000")
                .await
                .is_some(),
            "timed-out acquisition must not leak either node"
        );
    }

    #[tokio::test]
    async fn unrelated_release_storm_cannot_extend_the_retry_deadline() {
        let interval = Duration::from_millis(20);
        let gate = P2pNodeGate::new_isolated(interval, 3);
        let active = gate
            .acquire("http://worker-a:30000", "http://worker-b:30000")
            .await
            .expect("active pair must enter");
        let stop = Arc::new(AtomicBool::new(false));
        let storm = tokio::spawn({
            let gate = gate.clone();
            let stop = Arc::clone(&stop);
            async move {
                while !stop.load(Ordering::Acquire) {
                    let lease = gate
                        .acquire("http://worker-c:30000", "http://worker-d:30000")
                        .await
                        .expect("unrelated pair must remain available");
                    drop(lease);
                    tokio::task::yield_now().await;
                }
            }
        });
        let started = Instant::now();

        assert!(gate
            .acquire("http://worker-a:30000", "http://worker-e:30000")
            .await
            .is_none());
        let elapsed = started.elapsed();
        assert!(elapsed >= interval * 3);
        assert!(
            elapsed < Duration::from_millis(250),
            "release notifications must not postpone the absolute retry clock: {elapsed:?}"
        );

        stop.store(true, Ordering::Release);
        storm.await.unwrap();
        drop(active);
    }

    #[tokio::test]
    async fn cancelling_a_waiter_does_not_leak_a_node() {
        let gate = P2pNodeGate::new_isolated(Duration::from_secs(1), 1);
        let active = gate
            .acquire("http://worker-b:30000", "http://worker-c:30000")
            .await
            .expect("active pair must enter");
        let waiter = tokio::spawn({
            let gate = gate.clone();
            async move {
                gate.acquire("http://worker-a:30000", "http://worker-b:30000")
                    .await
            }
        });
        sleep(Duration::from_millis(20)).await;
        waiter.abort();
        assert!(waiter.await.is_err(), "waiter must be cancelled");

        assert!(
            gate.acquire("http://worker-a:30000", "http://worker-d:30000")
                .await
                .is_some(),
            "cancelled waiter must not retain its free endpoint"
        );
        drop(active);
    }

    #[tokio::test]
    async fn aborting_an_active_owner_releases_both_nodes_and_wakes_waiters() {
        let gate = P2pNodeGate::new_isolated(Duration::from_secs(1), 1);
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let owner = tokio::spawn({
            let gate = gate.clone();
            async move {
                let _lease = gate
                    .acquire("http://worker-a:30000", "http://worker-b:30000")
                    .await
                    .expect("owner must enter");
                entered_tx.send(()).unwrap();
                std::future::pending::<()>().await;
            }
        });
        entered_rx.await.expect("owner must report entry");

        let waiter = tokio::spawn({
            let gate = gate.clone();
            async move {
                gate.acquire("http://worker-a:30000", "http://worker-c:30000")
                    .await
            }
        });
        sleep(Duration::from_millis(20)).await;
        assert!(!waiter.is_finished());

        owner.abort();
        assert!(owner.await.is_err());
        assert!(timeout(Duration::from_millis(100), waiter)
            .await
            .expect("owner Drop must wake the waiter immediately")
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn broadcast_loser_rearms_for_the_next_release() {
        let gate = P2pNodeGate::new_isolated(Duration::from_secs(1), 1);
        let active = gate
            .acquire("http://worker-a:30000", "http://worker-b:30000")
            .await
            .expect("active pair must enter");
        let (entered_tx, mut entered_rx) = tokio::sync::mpsc::unbounded_channel();
        let release_first = Arc::new(Notify::new());
        let mut waiters = Vec::new();
        for target in ["worker-c", "worker-d"] {
            let gate = gate.clone();
            let entered_tx = entered_tx.clone();
            let release_first = Arc::clone(&release_first);
            waiters.push(tokio::spawn(async move {
                let _lease = gate
                    .acquire("http://worker-a:30000", &format!("http://{target}:30000"))
                    .await
                    .expect("waiter must eventually enter");
                entered_tx.send(target).unwrap();
                release_first.notified().await;
            }));
        }
        drop(entered_tx);
        sleep(Duration::from_millis(20)).await;
        drop(active);

        let first = timeout(Duration::from_millis(100), entered_rx.recv())
            .await
            .expect("one broadcast waiter must enter")
            .expect("entry channel must remain open");
        assert!(
            timeout(Duration::from_millis(20), entered_rx.recv())
                .await
                .is_err(),
            "only one waiter may own the shared node"
        );

        release_first.notify_one();
        let second = timeout(Duration::from_millis(100), entered_rx.recv())
            .await
            .expect("the losing waiter must re-arm and observe the next release")
            .expect("entry channel must remain open");
        assert_ne!(first, second);
        release_first.notify_one();
        for waiter in waiters {
            waiter.await.unwrap();
        }
    }

    #[tokio::test]
    async fn simultaneous_pair_race_never_double_owns_a_node() {
        const TASKS: usize = 24;
        let gate = P2pNodeGate::new_isolated(Duration::from_secs(1), 1);
        let barrier = Arc::new(tokio::sync::Barrier::new(TASKS + 1));
        let active_on_a = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for id in 0..TASKS {
            let gate = gate.clone();
            let barrier = Arc::clone(&barrier);
            let active_on_a = Arc::clone(&active_on_a);
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                let target = if id % 2 == 0 {
                    "http://worker-b:30000"
                } else {
                    "http://worker-c:30000"
                };
                let _lease = gate
                    .acquire("http://worker-a:30000", target)
                    .await
                    .expect("every short owner must enter before the retry deadline");
                assert_eq!(
                    active_on_a.fetch_add(1, Ordering::AcqRel),
                    0,
                    "node A must have only one owner"
                );
                sleep(Duration::from_millis(1)).await;
                assert_eq!(active_on_a.fetch_sub(1, Ordering::AcqRel), 1);
            }));
        }
        barrier.wait().await;
        for task in tasks {
            task.await.unwrap();
        }
    }

    #[tokio::test]
    async fn canonical_origins_share_one_node_lock() {
        let gate = P2pNodeGate::new_isolated(Duration::from_millis(20), 0);
        let active = gate
            .acquire("HTTP://WORKER-A:80/path", "http://worker-b:30000/")
            .await
            .expect("active pair must enter");

        assert!(
            gate.acquire("http://worker-a/", "http://worker-c:30000")
                .await
                .is_none(),
            "URL aliases for the same origin must share one node lock"
        );
        drop(active);
    }

    #[tokio::test]
    async fn router_instances_in_one_process_share_node_ownership() {
        let first = P2pNodeGate::new(Duration::from_millis(20), 0);
        let second = P2pNodeGate::new(Duration::from_millis(20), 0);
        let active = first
            .acquire(
                "http://process-shared-worker-a:30000",
                "http://process-shared-worker-b:30000",
            )
            .await
            .expect("first Router instance must enter");

        assert!(
            second
                .acquire(
                    "http://process-shared-worker-a:30000",
                    "http://process-shared-worker-c:30000",
                )
                .await
                .is_none(),
            "Router instances in one process must share per-node locks"
        );
        drop(active);
    }
}

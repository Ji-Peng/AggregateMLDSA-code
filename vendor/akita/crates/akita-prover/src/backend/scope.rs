use akita_error::AkitaError;

/// Consumer-issued proof lifetime identifier.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProofScopeId(pub(crate) u128);

impl ProofScopeId {
    pub const fn from_raw(value: u128) -> Self {
        Self(value)
    }

    pub const fn raw(self) -> u128 {
        self.0
    }
}

/// Scope lifecycle implemented by an opaque prover consumer.
pub trait ProofScopeConsumer {
    /// Backend-issued authority for one admitted proof lifetime.
    type ProofSessionHandle: Send + Sync + 'static;

    /// Finish a successfully completed scope.
    fn finish_scope(&self, session: &Self::ProofSessionHandle) -> Result<(), AkitaError>;

    /// Reclaim scope-owned resources after an error or early return.
    fn abort_scope_best_effort(&self, session: &Self::ProofSessionHandle);
}

/// RAII guard ensuring rejected or incomplete proofs trigger cleanup.
pub struct ProofScope<'a, Consumer: ProofScopeConsumer> {
    consumer: &'a Consumer,
    // `None` only after the caller has moved the proof lifetime out to resume
    // it later, for example across a scheduling boundary.
    session: Option<Consumer::ProofSessionHandle>,
    finished: bool,
}

impl<'a, Consumer: ProofScopeConsumer> ProofScope<'a, Consumer> {
    /// Guard a scope already validated by the backend's admission operation.
    pub fn admitted(consumer: &'a Consumer, session: Consumer::ProofSessionHandle) -> Self {
        Self {
            consumer,
            session: Some(session),
            finished: false,
        }
    }

    /// Borrow the authority for this admitted proof lifetime.
    pub fn session(&self) -> &Consumer::ProofSessionHandle {
        // The scope is admitted with a session and only `into_session` removes it.
        self.session
            .as_ref()
            .expect("admitted proof scope owns its session")
    }

    /// Move the admitted proof lifetime out so another scope can resume it.
    ///
    /// Dropping the returned handle without rebuilding a scope aborts the
    /// proof lifetime, exactly as dropping the original scope would.
    pub fn into_session(mut self) -> Consumer::ProofSessionHandle {
        let session = self
            .session
            .take()
            .expect("admitted proof scope owns its session");
        self.finished = true;
        session
    }

    /// Mark the scope complete and release consumer bookkeeping.
    pub fn finish(mut self) -> Result<(), AkitaError> {
        let session = self
            .session
            .take()
            .expect("admitted proof scope owns its session");
        self.consumer.finish_scope(&session)?;
        self.finished = true;
        Ok(())
    }
}

impl<Consumer: ProofScopeConsumer> Drop for ProofScope<'_, Consumer> {
    fn drop(&mut self) {
        if !self.finished {
            if let Some(session) = self.session.take() {
                self.consumer.abort_scope_best_effort(&session);
            }
        }
    }
}

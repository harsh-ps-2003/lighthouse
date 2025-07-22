mod fork_choice;
mod fork_choice_store;
mod metrics;

#[cfg(feature = "fast_confirmation")]
pub mod fast_confirmation;

pub use crate::fork_choice::{
    AttestationFromBlock, Error, ForkChoice, ForkChoiceView, ForkchoiceUpdateParameters,
    InvalidAttestation, InvalidBlock, PayloadVerificationStatus, PersistedForkChoice,
    QueuedAttestation, ResetPayloadStatuses,
};
pub use fork_choice_store::ForkChoiceStore;
pub use proto_array::{
    Block as ProtoBlock, ExecutionStatus, InvalidationOperation, ProposerHeadError,
};

#[cfg(feature = "fast_confirmation")]
pub use crate::fast_confirmation::{FastConfirmation, FastConfirmationConfig, FcrMeta};

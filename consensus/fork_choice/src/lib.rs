mod fork_choice;
mod fork_choice_store;
mod metrics;

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

pub use crate::fast_confirmation::{
    FastConfirmation, FastConfirmationConfig, FcrMeta, StateProvider,
    DEFAULT_FCR_BYZANTINE_THRESHOLD_PERCENTAGE, DEFAULT_FCR_SLASHING_THRESHOLD_PERCENTAGE,
};

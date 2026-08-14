use super::config::{ActiveAmxConfig, ACTIVE_CONFIG, SESSION_DEPTH};
use super::raw;
use super::AmxConfig;

/// A session guard that manages AMX tile configuration lifecycle on the current thread.
pub struct AmxSession {
    _private: (),
}

/// Error returned when an AMX session cannot be entered safely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmxSessionError {
    /// The current host cannot execute AMX tile instructions safely.
    UnsupportedTarget,
}

impl core::fmt::Display for AmxSessionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedTarget => write!(
                f,
                "AMX tile instructions are not supported or enabled for this process"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AmxSessionError {}

impl AmxSession {
    /// Returns true if an AMX session is currently active on the executing thread.
    #[inline]
    pub fn is_active() -> bool {
        ACTIVE_CONFIG.with(|c| c.get().is_some())
    }

    /// Enter a new AMX compute phase with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`AmxSessionError::UnsupportedTarget`] when AMX is not supported
    /// or enabled for the current process.
    #[inline]
    pub fn new(config: &AmxConfig) -> Result<Self, AmxSessionError> {
        if !super::amx_runtime_supported() {
            return Err(AmxSessionError::UnsupportedTarget);
        }

        let depth = SESSION_DEPTH.with(|d| {
            let val = d.get();
            d.set(val + 1);
            val
        });

        if depth == 0 {
            unsafe {
                raw::ldtilecfg(config);
            }
            ACTIVE_CONFIG.with(|c| c.set(Some(ActiveAmxConfig::from(config))));
        } else {
            let active = ACTIVE_CONFIG.with(|c| c.get());
            if active != Some(ActiveAmxConfig::from(config)) {
                unsafe {
                    raw::ldtilecfg(config);
                }
                ACTIVE_CONFIG.with(|c| c.set(Some(ActiveAmxConfig::from(config))));
            }
        }
        Ok(Self { _private: () })
    }

    /// Context switch mitigation: release tile registers explicitly.
    #[inline]
    pub fn release() {
        let active = Self::is_active();
        if active && super::amx_runtime_supported() {
            unsafe {
                raw::tilerelease();
            }
        }
        ACTIVE_CONFIG.with(|c| c.set(None));
        SESSION_DEPTH.with(|d| d.set(0));
    }
}

impl Drop for AmxSession {
    #[inline]
    fn drop(&mut self) {
        let depth = SESSION_DEPTH.with(|d| {
            let val = d.get();
            if val > 0 {
                d.set(val - 1);
                val - 1
            } else {
                0
            }
        });

        if depth == 0 {
            unsafe {
                raw::tilerelease();
            }
            ACTIVE_CONFIG.with(|c| c.set(None));
        }
    }
}

/// An RAII guard that encapsulates a complete AMX batch computation.
///
/// Automatically releases the AMX registers (`tilerelease()`) when dropped to prevent
/// context-switch penalties.
pub struct AmxBatchSession;

impl AmxBatchSession {
    /// Begin a new AMX batch computation.
    ///
    /// # Errors
    ///
    /// Returns [`AmxSessionError::UnsupportedTarget`] when AMX is not supported
    /// or enabled for the current process.
    #[inline]
    pub fn begin(config: &AmxConfig) -> Result<Self, AmxSessionError> {
        if !super::amx_runtime_supported() {
            return Err(AmxSessionError::UnsupportedTarget);
        }

        unsafe {
            raw::ldtilecfg(config);
        }
        ACTIVE_CONFIG.with(|c| c.set(Some(ActiveAmxConfig::from(config))));
        Ok(Self)
    }
}

impl Drop for AmxBatchSession {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            raw::tilerelease();
        }
        ACTIVE_CONFIG.with(|c| c.set(None));
        SESSION_DEPTH.with(|d| d.set(0));
    }
}

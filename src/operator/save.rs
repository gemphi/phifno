//! Saving and loading an operator: the versioned format, with or without the optimiser.

mod compat;
use compat::{saved_compact, saved_v1, saved_v2};
use super::{Fno, FnoConfig};
use phistd::{Adam, AdamState, Tensor};
use phistd::c64;
use serde::{Deserialize, Serialize};

pub(super) type SavedParam = ([usize; 3], Vec<(f64, f64)>);
/// A parameter in the compact format: its shape and `f32` pairs.
pub(super) type CompactParam = ([usize; 3], Vec<(f32, f32)>);

/// Functions per forward pass in [`Fno::apply`].
pub const APPLY_CHUNK: usize = 256;

/// File header: magic and format version. Version 1 had no header.
const MAGIC: &[u8] = b"PHINET";
const FORMAT_VERSION: u16 = 2;
/// Format 3: the same, parameters as `f32` pairs, no optimiser state.
const FORMAT_VERSION_COMPACT: u16 = 3;

#[derive(Serialize, Deserialize)]
pub(super) struct SavedCompact {
    pub(super) config: FnoConfig,
    pub(super) params: Vec<CompactParam>,
}

#[derive(Serialize, Deserialize)]
pub(super) struct SavedV1 {
    pub(super) config: FnoConfig,
    pub(super) params: Vec<SavedParam>,
}

#[derive(Serialize, Deserialize)]
pub(super) struct Saved {
    pub(super) config: FnoConfig,
    pub(super) params: Vec<SavedParam>,
    pub(super) adam: Option<AdamState>,
}

impl Fno {
    pub(super) fn saved_params(&self) -> Vec<SavedParam> {
        self.params()
            .iter()
            .map(|p| {
                let t = self.tape.value(*p);
                (t.shape, t.data.iter().map(|z| (z.re, z.im)).collect())
            })
            .collect()
    }

    /// Serialises the configuration and parameters: the magic `PHINET`, a
    /// format version (`2`), then bincode of `Saved`. Optimiser state is not
    /// included; see [`Fno::to_bytes_with`].
    pub fn to_bytes(&self) -> Vec<u8> {
        self.encode(None)
    }

    /// [`Fno::to_bytes`] plus the optimiser's moments and step count, so a run
    /// resumes exactly where it stopped.
    pub fn to_bytes_with(&self, adam: &Adam) -> Vec<u8> {
        self.encode(Some(adam.state()))
    }

    /// [`Fno::to_bytes`] with the parameters stored as `f32` pairs (format
    /// version 3): half the size, about seven significant digits, the
    /// computation stays `f64` after loading. `tests/save.rs` bounds the
    /// drift a round trip causes on `apply`.
    pub fn to_bytes_compact(&self) -> Vec<u8> {
        let saved = SavedCompact {
            config: self.config,
            params: self
                .params()
                .iter()
                .map(|p| {
                    let t = self.tape.value(*p);
                    (t.shape, t.data.iter().map(|z| (z.re as f32, z.im as f32)).collect())
                })
                .collect(),
        };
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION_COMPACT.to_le_bytes());
        out.extend(bincode::serialize(&saved).unwrap_or_default());
        out
    }

    fn encode(&self, adam: Option<AdamState>) -> Vec<u8> {
        let saved = Saved { config: self.config, params: self.saved_params(), adam };
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend(bincode::serialize(&saved).unwrap_or_default());
        out
    }

    /// Loads an operator written by [`Fno::to_bytes`] or [`Fno::to_bytes_with`]
    /// (the optimiser state, if any, is dropped) — or by the pre-version
    /// format, which carried no header.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        Self::from_bytes_with(bytes).map(|(f, _)| f)
    }

    /// Loads an operator and, when the file carried one, its optimiser.
    pub fn from_bytes_with(bytes: &[u8]) -> Option<(Self, Option<Adam>)> {
        let saved: Saved = if bytes.starts_with(MAGIC) {
            if bytes.len() < MAGIC.len() + 2 {
                return None;
            }
            let v = u16::from_le_bytes([bytes[MAGIC.len()], bytes[MAGIC.len() + 1]]);
            let body = &bytes[MAGIC.len() + 2..];
            match v {
                FORMAT_VERSION => saved_v2(body)?,
                FORMAT_VERSION_COMPACT => {
                    let (config, params) = saved_compact(body)?;
                    Saved {
                        config,
                        params: params.into_iter().map(|(sh, d)| (sh, d.into_iter().map(|(re, im)| (re as f64, im as f64)).collect())).collect(),
                        adam: None,
                    }
                }
                _ => return None,
            }
        } else {
            // Format 1: bincode of (config, params) with no header.
            let legacy = saved_v1(bytes)?;
            Saved { config: legacy.config, params: legacy.params, adam: None }
        };
        let mut fno = Self::new(saved.config, 0);
        let params = fno.params();
        if params.len() != saved.params.len() {
            return None;
        }
        for (p, (shape, data)) in params.iter().zip(saved.params) {
            let t = Tensor::from_vec(shape, data.into_iter().map(|(re, im)| c64::new(re, im)).collect());
            fno.tape.set_value(*p, t);
        }
        Some((fno, saved.adam.map(Adam::from_state)))
    }
}

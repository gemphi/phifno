//! Loading operators saved before `FnoConfig::tie` existed.
use super::{CompactParam, Saved, SavedCompact, SavedParam, SavedV1};
use super::super::FnoConfig;
use phistd::AdamState;
use serde::{Deserialize, Serialize};

/// The configuration as it was serialised before `tie` existed. bincode is
/// not self-describing, so `#[serde(default)]` cannot rescue an old file;
/// every load tries the current layout first and this one second, and an
/// old file is a circle operator by construction.
#[derive(Serialize, Deserialize)]
struct FnoConfigNoTie {
    space: usize,
    width: usize,
    layers: usize,
    modes: usize,
    margin: f64,
    lr: f64,
}

impl From<FnoConfigNoTie> for FnoConfig {
    fn from(c: FnoConfigNoTie) -> Self {
        FnoConfig { space: c.space, width: c.width, layers: c.layers, modes: c.modes, margin: c.margin, lr: c.lr, tie: super::super::Tie::Circle }
    }
}

#[derive(Deserialize)]
struct SavedNoTie {
    config: FnoConfigNoTie,
    params: Vec<SavedParam>,
    adam: Option<AdamState>,
}

#[derive(Deserialize)]
struct SavedCompactNoTie {
    config: FnoConfigNoTie,
    params: Vec<CompactParam>,
}

#[derive(Deserialize)]
struct SavedV1NoTie {
    config: FnoConfigNoTie,
    params: Vec<SavedParam>,
}

pub(super) fn saved_v2(body: &[u8]) -> Option<Saved> {
    bincode::deserialize::<Saved>(body).ok().or_else(|| {
        let s: SavedNoTie = bincode::deserialize(body).ok()?;
        Some(Saved { config: s.config.into(), params: s.params, adam: s.adam })
    })
}

pub(super) fn saved_compact(body: &[u8]) -> Option<(FnoConfig, Vec<CompactParam>)> {
    bincode::deserialize::<SavedCompact>(body).ok().map(|c| (c.config, c.params)).or_else(|| {
        let c: SavedCompactNoTie = bincode::deserialize(body).ok()?;
        Some((c.config.into(), c.params))
    })
}

pub(super) fn saved_v1(bytes: &[u8]) -> Option<SavedV1> {
    bincode::deserialize::<SavedV1>(bytes).ok().or_else(|| {
        let s: SavedV1NoTie = bincode::deserialize(bytes).ok()?;
        Some(SavedV1 { config: s.config.into(), params: s.params })
    })
}

//! The `Module` trait and a two-operator stack on one tape.

use super::Fno;
use phistd::{Tape, Var};

/// Anything with parameters and a forward pass on a tape. `Fno` implements
/// it; a stack of operators or a wrapper that adds a table does too.
pub trait Module {
    /// The tape the module's parameters live on.
    fn tape(&mut self) -> &mut Tape;
    /// All trainable parameters, in a stable order.
    fn parameters(&self) -> Vec<Var>;
    /// Forward pass on a value already on the tape.
    fn forward(&mut self, x: Var) -> Var;
    /// Number of complex parameters.
    fn parameter_count(&self) -> usize;
}

impl Module for Fno {
    fn tape(&mut self) -> &mut Tape {
        &mut self.tape
    }
    fn parameters(&self) -> Vec<Var> {
        self.params()
    }
    fn forward(&mut self, x: Var) -> Var {
        Fno::forward(self, x)
    }
    fn parameter_count(&self) -> usize {
        Fno::parameter_count(self)
    }
}

/// Two modules applied in sequence on one tape — the second's parameters are
/// attached to the first's tape at construction.
pub struct Sequential {
    pub first: Fno,
    second_params: Vec<Var>,
    second: Fno,
}

impl Sequential {
    /// Composes `first` then `second`; `second`'s parameters are copied onto
    /// `first`'s tape so one `backward` reaches both.
    pub fn new(mut first: Fno, second: Fno) -> Self {
        let second_params: Vec<Var> = second.params().iter().map(|p| first.attach_param(second.tape.value(*p).clone())).collect();
        Self { first, second_params, second }
    }
}

impl Module for Sequential {
    fn tape(&mut self) -> &mut Tape {
        &mut self.first.tape
    }
    fn parameters(&self) -> Vec<Var> {
        let mut v = self.first.params();
        v.extend(self.second_params.iter().copied());
        v
    }
    fn forward(&mut self, x: Var) -> Var {
        let h = self.first.forward(x);
        // The second operator's graph, with its parameters as they live on
        // the first tape: same layers and tie, different Vars.
        super::graph::forward(&mut self.first.tape, &self.second.config, &self.second_params, h)
    }
    fn parameter_count(&self) -> usize {
        self.parameters().iter().map(|p| self.first.tape.value(*p).len()).sum()
    }
}

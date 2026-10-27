# phifno — Phi Fourier Neural Operators

In Phient's world, a word (w) is a seed: what it relates to, under what conditions (c), grows from it. The Phical paradigm, X ⇒ Y, drives and composes Phient's Phasor Language Models: Phinet, Phiano, PhiTon. Phical stores no point: a word is 32–64 relations, each a complex phasor on a closed circle; meaning is a side effect of pairs of words moving under a condition, and they move by the error of what was expected — toward what came, away from what did not.

phifno is the spherical Fourier neural operator, for words: learning in the frequency domain, where a relation is not a weight in a matrix but a mode on the sphere. The operator half (`operator/`: `Fno`, `FnoConfig`, `Tie`) learns mappings between phasor states; the sphere half (`sphere/`) carries the spherical harmonics; beside it, `heat.rs` is the heat flow that smooths what is learned across the circle rather than across a grid.

In the [phiano arena](https://github.com/gemphi/phiano) it stands as its own reader (the `Operator` arm, batch-trained at the split), scored on the same stream as the counts, the concepts, the conjugate mirror, and the phasor engine — one run, one table, so a claim about which model is learning better is a measurement, not an opinion.

Part of the [phinet](https://github.com/gemphi/phinet) workspace. MIT OR Apache-2.0.

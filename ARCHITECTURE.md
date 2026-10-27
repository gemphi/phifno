# phifno — architecture

## The crate, module by module

```mermaid
graph TD
    librs["lib.rs<br/>Word · pub mod fft · pub use"]
    subgraph operator_["operator/ — the Fourier neural operator"]
        omod["mod.rs<br/>FnoConfig · Tie · Example · MixtureExample · Layer"]
        fnors["fno.rs · resolution.rs<br/>Fno · new · forward · apply · at_resolution"]
        fwdrs["graph.rs<br/>forward: lift · layers · projection"]
        modesrs["modes.rs<br/>mask_modes · scale_modes"]
        modulers["module.rs<br/>Module · Sequential"]
        samplerrs["sampler.rs<br/>FrequencySampler · zeta_weights · rank"]
        trainrs["train.rs<br/>train_epoch · _with · _mixture · _softmax<br/>decoys from phistd::decoy"]
        savers["save.rs<br/>to_bytes · from_bytes · APPLY_CHUNK"]
        compatrs["save/compat.rs<br/>layouts before Tie"]
        symmetryrs["symmetry.rs<br/>defects"]
        otests["tests.rs<br/>tests/fno · tests/save · tests/sphere"]
    end
    subgraph sphere_["sphere/ — the 64 real harmonics"]
        smod["mod.rs<br/>degree_gate · degree_energy"]
        basisrs["basis.rs<br/>DEGREES · HARMONICS · real_harmonic"]
        gridrs["grid.rs<br/>Grid · gauss_legendre"]
        transformrs["transform.rs<br/>Sphere · synthesis · analysis"]
        stests["tests.rs"]
    end
    heatrs["heat.rs<br/>circle · sphere · sphere_coeffs · overlap_after"]
    phistd["phistd<br/>Tape · Tensor · Var · c64 · Function · Adam · fft"]
    subgraph users["workspace crates that depend on phifno"]
        phical["phical"]
        phiano["phiano"]
    end

    librs --> omod & smod & heatrs
    librs -->|"Word"| smod
    librs -->|"mod fft"| phistd
    omod --> fnors & fwdrs & modesrs & modulers & samplerrs & trainrs & savers & symmetryrs & otests
    fnors -->|"forward"| fwdrs
    fnors -->|"APPLY_CHUNK"| savers
    fnors -->|"at_resolution"| modesrs
    modulers -->|"Sequential second operator"| fwdrs
    modulers --> fnors
    trainrs -->|"mask_modes"| modesrs
    trainrs --> fnors
    samplerrs -->|"Example"| omod
    savers --> compatrs
    symmetryrs -->|"apply"| fnors
    omod -.->|"HARMONICS · DEGREES via crate root"| basisrs
    otests -.->|"tests/sphere"| smod
    smod --> basisrs & gridrs & transformrs & stests
    transformrs --> basisrs & gridrs
    operator_ --> phistd
    sphere_ --> phistd
    phical & phiano -.-> librs
```

## The module pages

```mermaid
graph LR
    A["ARCHITECTURE.md"] --> L["lib.md<br/>lib.rs · Word · fft"]
    A --> O["operator.md<br/>operator/"]
    A --> S["sphere.md<br/>sphere/"]
    A --> H["heat.md<br/>heat.rs"]
```

| diagram | file |
|---|---|
| the crate root, Word, fft | [docs/phifno/lib.md](../../docs/phifno/lib.md) |
| the Fourier neural operator | [docs/phifno/operator.md](../../docs/phifno/operator.md) |
| the sphere and its harmonics | [docs/phifno/sphere.md](../../docs/phifno/sphere.md) |
| heat flow on the circle | [docs/phifno/heat.md](../../docs/phifno/heat.md) |

## Status

2026-10-27 · 24 test functions · what is left: [TODO.md](TODO.md)

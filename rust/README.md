# twisterl-rs

`twisterl-rs` provides the Rust core for the [TwisteRL](https://github.com/IBM/twisteRL) project. It implements reinforcement learning primitives and a Python extension for high performance training and inference.

## Features

- Generic `Env` trait for implementing discrete environments
- Sample puzzle environment
- Simple neural network layers and policy utilities
- Parallel collectors for PPO and AlphaZero algorithms
- Optional Python bindings via [PyO3](https://pyo3.rs)

This crate is currently a proof of concept and its API may change.

## Usage
Add the crate to your `Cargo.toml`:

```toml
[dependencies]
twisterl-rs = { version = "0.1"}
```

## Python interface

When compiled with the `python_bindings` feature the crate exposes bindings used by the `twisterl` Python package. See the [repository](https://github.com/IBM/twisteRL)  for Python usage examples.

In this case, add the crate to your `Cargo.toml` with the `python_bindings` feature:

```toml
[dependencies]
twisterl-rs = { version = "0.1", features = ["python_bindings"]}
```

## License

Licensed under the [Apache License, Version 2.0.](https://github.com/dtolnay/syn/blob/HEAD/LICENSE-APACHE).
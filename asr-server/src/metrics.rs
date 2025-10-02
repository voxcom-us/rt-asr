// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use lazy_static::lazy_static;
use prometheus::{
    histogram_opts, labels, opts, register_counter, register_gauge, register_histogram,
};
use prometheus::{Counter, Gauge, Histogram};

pub mod asr {
    use super::*;
    lazy_static! {
        pub static ref CONNECT: Counter = register_counter!(opts!(
            "asr_connect",
            "Number of connections to the asr.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref MODEL_STEP_DURATION: Histogram = register_histogram!(histogram_opts!(
            "asr_model_step_duration",
            "ASR model step duration distribution.",
            vec![20e-3, 30e-3, 40e-3, 50e-3, 60e-3, 70e-3, 80e-3],
        ))
        .unwrap();
        pub static ref CONNECTION_NUM_STEPS: Histogram = register_histogram!(histogram_opts!(
            "asr_connection_num_steps",
            "ASR model, distribution of number of steps for a connection.",
            vec![2., 25., 125., 250., 500., 750., 1125., 1500., 2250., 3000., 4500.],
        ))
        .unwrap();
        pub static ref OPEN_CHANNELS: Gauge = register_gauge!(opts!(
            "asr_open_channels",
            "Number of open channels (users currently connected).",
            labels! {"handler" => "all",}
        ))
        .unwrap();
    }
}

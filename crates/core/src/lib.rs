//! core
mod sampler;

// #[global_allocator]
// static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
use log::info;
use nuts_rs::CpuLogpFunc;
use sampler::{be_nuts, MyDivergenceInfo};

/// A single chain run.
#[derive(Default)]
pub struct Run {}

impl Run {
    fn run<M>(
        &self,
        model: M,
        initial_position: &[f64],
        chain: u64,
        seed: u64,
        tuning: u64,
        samples: u64,
    ) -> ChainRun
    where
        M: CpuLogpFunc,
    {
        info!("chain={chain} seed={seed}");
        let (trace, stats) = be_nuts(model, initial_position, chain, tuning, samples, seed);

        ChainRun { trace, stats }
    }
}

/// A single chain run.
pub struct ChainRun {
    /// The trace for the chain.
    pub trace: Vec<Box<[f64]>>,
    /// The stats for divergences.
    pub stats: Vec<MyDivergenceInfo>,
}

impl ChainRun {
    /// Return the trace for a given parameter.
    pub fn trace(&self, i: usize) -> Vec<f64> {
        self.trace.iter().map(|x| x[i]).collect::<Vec<_>>()
    }

    /// Return the stats for divergences.
    #[allow(dead_code)]
    pub fn stats(&self) -> &Vec<MyDivergenceInfo> {
        &self.stats
    }
}

/// A collection of chains
pub struct Chains {
    /// The chains.
    pub chains: Vec<ChainRun>,
    /// The dimensionality of the model.
    pub dim: usize,
    /// The parameters of the model.
    pub parameters: Vec<String>,
}

/// A model.
pub trait Model: CpuLogpFunc {
    /// The parameters of the model.
    fn parameters(&self) -> Vec<String>;
}

impl Chains {
    /// Runs a collection of chains - sequentially.
    pub fn run<M: Model>(
        seed: u64,
        initial_position: &[f64],
        model: M,
        chain_count: u64,
        tuning: u64,
        samples: u64,
    ) -> Self
    where
        M: CpuLogpFunc + Clone,
    {
        let chains = (0..chain_count)
            .map(|chain| {
                Run::default().run(
                    model.clone(),
                    initial_position,
                    chain,
                    seed + chain,
                    tuning,
                    samples,
                )
            })
            .collect();

        Chains {
            chains,
            dim: model.dim(),
            parameters: model.parameters(),
        }
    }
}

impl Chains {
    /// Returns the extrema for a given parameter - across all chains.
    pub fn extrema(&self, i: usize) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for chain in &self.chains {
            let (min_, max_) = chain
                .trace(i)
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), x| {
                    (min.min(*x), max.max(*x))
                });

            min = min.min(min_);
            max = max.max(max_);
        }

        (min, max)
    }

    /// Returns the traces for a given parameter
    pub fn traces(&self, i: usize) -> Vec<Vec<f64>> {
        self.chains.iter().map(|x| x.trace(i)).collect()
    }
}

/// Runs a collection of chains - sequentially.
pub fn run_with<M: Model + Clone>(
    seed: u64,
    initial_position: &[f64],
    model: M,
    chain_count: u64,
    tuning: u64,
    samples: u64,
) -> Chains {
    Chains::run(seed, initial_position, model, chain_count, tuning, samples)
}

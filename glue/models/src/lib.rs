//! nextcuda-models: the models a machine serves (the registry), the settings file, and the catalog of supported
//! models. A library, so another program can install and list models the way `nextcuda models` does.

pub mod catalog;
pub mod config;
pub mod registry;

pub use config::Config;

//! A kind's server as a plain process, not a container. nextsycl ran every server in a container because the SYCL
//! runtime (oneAPI) and the GPU groups came with the image; on an NVIDIA box with the driver and CUDA on the host the
//! program runs as it is. Set `NS_CONTAINER_ENGINE` to a container program to get the container mode back.
//!
//! A server is `<this program> <kind> serve ID ... --here` started detached; its pid and port are files in the
//! socket directory (`<name>.pid`, `<name>.port`), its log `<name>.log`. `running`, `stop` and the address are read
//! from those, so `ps`, `stop` and `logs` behave as they do for a container.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use nextcuda_models::Config;

/// Whether servers run as processes here (no container program was asked for)
pub fn native(cfg: &Config) -> bool {
    cfg.get("NS_CONTAINER_ENGINE").is_none()
}

fn dir(cfg: &Config) -> PathBuf {
    cfg.socket_dir()
}

fn file(cfg: &Config, name: &str, ext: &str) -> PathBuf {
    dir(cfg).join(format!("{name}.{ext}"))
}

fn pid(cfg: &Config, name: &str) -> Option<u32> {
    std::fs::read_to_string(file(cfg, name, "pid")).ok()?.trim().parse().ok()
}

/// A process with `pid` is there (and is not a zombie)
fn alive(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|s| !s.rsplit(')').next().unwrap_or("").trim_start().starts_with('Z'))
}

pub fn running(cfg: &Config, name: &str) -> bool {
    pid(cfg, name).is_some_and(alive)
}

pub fn port(cfg: &Config, name: &str) -> Option<String> {
    std::fs::read_to_string(file(cfg, name, "port")).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

pub fn log_path(cfg: &Config, name: &str) -> PathBuf {
    file(cfg, name, "log")
}

/// `exe args` as a detached process named `name`, listening on `port`; its output goes to the log. `foreground`: run
/// it here and return when it ends.
pub fn launch(cfg: &Config, name: &str, port: &str, exe: &std::path::Path, args: &[String], foreground: bool) -> Result<(), String> {
    std::fs::create_dir_all(dir(cfg)).map_err(|e| format!("{}: {e}", dir(cfg).display()))?;
    if running(cfg, name) {
        return Err(format!("{name} is running already ({} stop)", exe.file_name().map_or("nextcuda".into(), |n| n.to_string_lossy())));
    }
    let _ = std::fs::write(file(cfg, name, "port"), port);
    let mut c = Command::new(exe);
    c.args(args).stdin(Stdio::null());
    if foreground {
        let mut ch = c.spawn().map_err(|e| format!("{}: {e}", exe.display()))?;
        let _ = std::fs::write(file(cfg, name, "pid"), ch.id().to_string());
        let st = ch.wait().map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(file(cfg, name, "pid"));
        return if st.success() { Ok(()) } else { Err(format!("{name} ended ({st})")) };
    }
    let log = std::fs::OpenOptions::new().create(true).append(true).open(log_path(cfg, name)).map_err(|e| e.to_string())?;
    let err = log.try_clone().map_err(|e| e.to_string())?;
    c.stdout(Stdio::from(log)).stderr(Stdio::from(err));
    let ch = c.spawn().map_err(|e| format!("{}: {e}", exe.display()))?;
    std::fs::write(file(cfg, name, "pid"), ch.id().to_string()).map_err(|e| e.to_string())?;
    // the child outlives this process: it is not waited for
    std::mem::forget(ch);
    Ok(())
}

/// SIGTERM, then SIGKILL after `secs`
pub fn stop(cfg: &Config, name: &str, secs: u64) {
    let Some(p) = pid(cfg, name) else { return };
    let _ = Command::new("kill").arg(p.to_string()).status();
    let t0 = Instant::now();
    while alive(p) && t0.elapsed() < Duration::from_secs(secs) {
        std::thread::sleep(Duration::from_millis(200));
    }
    if alive(p) {
        let _ = Command::new("kill").args(["-9", &p.to_string()]).status();
    }
    let _ = std::fs::remove_file(file(cfg, name, "pid"));
}

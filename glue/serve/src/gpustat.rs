//! The GPU telemetry sampler (`nextcuda gpustat`): NVIDIA cards, read with `nvidia-smi`, written every few seconds
//! to a JSON file (default /run/gpustat.json) that pages read for their GPU pill and monitor - and sampled in
//! process by the front door, which keeps each card's free memory for placing models.
//!
//! nextsycl's sampler read the xe driver's sysfs and the DRM clients' fdinfo (the only VRAM accounting an Intel card
//! has). An NVIDIA driver answers all of it through one query, so this is that query, with the same fields in the
//! file: the name, VRAM used and total, busy %, temperatures, power and the cap, fan, clock, the PCIe link. The
//! card's energy is the integral of the power it drew between samples (the driver has no counter this query can
//! read); the VRAM each process holds is `--query-compute-apps`.

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

pub struct Options {
    pub out: PathBuf,
    pub interval: Duration,
    /// kept for the command line's sake: nvidia-smi's own power is instantaneous
    pub power_window: Duration,
    /// the card (PCI address); none: the first NVIDIA GPU
    pub pci: Option<String>,
    /// VRAM in MiB; none: what the driver says
    pub vram_mb: Option<u64>,
    /// one sample, printed instead of written
    pub once: bool,
}

const FIELDS: &str = "pci.bus_id,name,memory.used,memory.total,utilization.gpu,temperature.gpu,temperature.memory,power.draw,power.limit,fan.speed,clocks.sm,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max";

/// "00000000:09:00.0" as "0000:09:00.0" (sysfs's, lower case)
fn pci_of(bus_id: &str) -> String {
    let b = bus_id.trim().to_lowercase();
    match b.split_once(':') {
        Some((dom, rest)) if dom.len() > 4 => format!("{}:{rest}", &dom[dom.len() - 4..]),
        _ => b,
    }
}

fn smi(args: &[&str]) -> Option<String> {
    let o = Command::new("nvidia-smi").args(args).output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).into_owned())
}

/// One card's row of the query
#[derive(Clone, Debug, Default)]
struct Row {
    pci: String,
    name: String,
    used_mb: u64,
    total_mb: u64,
    busy: Option<f64>,
    temp: Option<f64>,
    temp_mem: Option<f64>,
    power: Option<f64>,
    cap: Option<f64>,
    fan_pct: Option<f64>,
    sm_mhz: Option<f64>,
    gen: Option<u32>,
    width: Option<u32>,
    gen_max: Option<u32>,
    width_max: Option<u32>,
}

fn rows() -> Vec<Row> {
    let Some(out) = smi(&[&format!("--query-gpu={FIELDS}"), "--format=csv,noheader,nounits"]) else { return Vec::new() };
    out.lines().filter_map(|l| {
        let f: Vec<&str> = l.split(',').map(str::trim).collect();
        if f.len() < 15 {
            return None;
        }
        let n = |i: usize| f[i].parse::<f64>().ok();
        Some(Row {
            pci: pci_of(f[0]),
            name: f[1].to_string(),
            used_mb: n(2).unwrap_or(0.0) as u64,
            total_mb: n(3).unwrap_or(0.0) as u64,
            busy: n(4),
            temp: n(5),
            temp_mem: n(6),
            power: n(7),
            cap: n(8),
            fan_pct: n(9),
            sm_mhz: n(10),
            gen: n(11).map(|v| v as u32),
            width: n(12).map(|v| v as u32),
            gen_max: n(13).map(|v| v as u32),
            width_max: n(14).map(|v| v as u32),
        })
    }).collect()
}

/// The first NVIDIA GPU, as a PCI address
pub fn find_card() -> Option<String> {
    find_cards().into_iter().next()
}

/// Every NVIDIA GPU by PCI address, in nvidia-smi's order (PCI bus order: the engines are started with
/// CUDA_DEVICE_ORDER=PCI_BUS_ID so a CUDA index is the same number)
pub fn find_cards() -> Vec<String> {
    rows().into_iter().map(|r| r.pci).collect()
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// One card's sampler
pub struct Card {
    pub pci: String,
    pub name: String,
    total_mb: u64,
    vram_override: Option<u64>,
    /// joules so far (power x time between samples)
    energy_j: f64,
    last: Option<Instant>,
}

impl Card {
    pub fn open(pci: &str, vram_mb: Option<u64>) -> Card {
        let r = rows().into_iter().find(|r| r.pci == pci);
        Card {
            pci: pci.to_string(),
            name: r.as_ref().map_or_else(|| format!("NVIDIA GPU {pci}"), |r| r.name.clone()),
            total_mb: vram_mb.or_else(|| r.as_ref().map(|r| r.total_mb)).unwrap_or(0),
            vram_override: vram_mb,
            energy_j: 0.0,
            last: None,
        }
    }

    /// A reading (the gpustat file's fields) and the VRAM each process holds on the card, in MiB
    pub fn sample(&mut self, _power_window: Duration) -> (Value, Vec<(u32, u64)>) {
        let now = Instant::now();
        let r = rows().into_iter().find(|r| r.pci == self.pci).unwrap_or_default();
        if let (Some(last), Some(p)) = (self.last, r.power) {
            self.energy_j += p * (now - last).as_secs_f64();
        }
        self.last = Some(now);
        if self.vram_override.is_none() && r.total_mb > 0 {
            self.total_mb = r.total_mb;
        }
        let load1 = std::fs::read_to_string("/proc/loadavg").ok().and_then(|l| l.split_whitespace().next()?.parse::<f64>().ok()).unwrap_or(0.0);
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
        let link = |g: Option<u32>, w: Option<u32>| json!({"gen": g, "width": w.unwrap_or(0)});
        let out = json!({
            "name": self.name, "pci": self.pci, "ts": ts,
            "vram_used_mb": r.used_mb, "vram_total_mb": self.total_mb,
            "busy_pct": r.busy.map(round1).unwrap_or(0.0),
            "temp_pkg": r.temp, "temp_vram": r.temp_mem,
            "power_w": r.power.map(round1), "pkg_power_w": Value::Null, "power_cap_w": r.cap.map(|c| c.round() as i64),
            "temp_vram_max": r.temp_mem, "temp_pcie": Value::Null, "temp_mctrl": Value::Null,
            "pcie": json!({"cur": link(r.gen, r.width), "card_max": link(r.gen_max, r.width_max), "slot_max": link(r.gen_max, r.width_max)}),
            "host_load1": load1, "host_cpus": std::thread::available_parallelism().map_or(1, |n| n.get()),
            "energy_j": round1(self.energy_j),
            "fan_pct": r.fan_pct, "fan_rpm": Value::Null,
            "freq_mhz": r.sm_mhz.map(|f| f.round() as i64),
        });
        let procs = smi(&["--query-compute-apps=pid,gpu_bus_id,used_memory", "--format=csv,noheader,nounits"])
            .map(|o| {
                let mut v: Vec<(u32, u64)> = o.lines().filter_map(|l| {
                    let f: Vec<&str> = l.split(',').map(str::trim).collect();
                    if f.len() < 3 || pci_of(f[1]) != self.pci {
                        return None;
                    }
                    Some((f[0].parse::<u32>().ok()?, f[2].parse::<u64>().ok()?))
                }).collect();
                v.sort_by_key(|p| std::cmp::Reverse(p.1));
                v
            })
            .unwrap_or_default();
        (out, procs)
    }
}

/// Samples until the process ends (or once)
pub fn run(o: Options) -> Result<(), String> {
    let pci = o.pci.clone().or_else(find_card).ok_or("no NVIDIA GPU (is nvidia-smi on PATH? --pci ADDR)")?;
    let mut card = Card::open(&pci, o.vram_mb);
    loop {
        let (out, _) = card.sample(o.power_window);
        if o.once {
            println!("{out}");
            return Ok(());
        }
        let tmp = o.out.with_extension("json.tmp");
        std::fs::write(&tmp, out.to_string()).and_then(|_| {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o644))?;
            std::fs::rename(&tmp, &o.out)
        }).map_err(|e| format!("{}: {e}", o.out.display()))?;
        std::thread::sleep(o.interval);
    }
}

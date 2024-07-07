extern crate kdam;

use kdam::{tqdm, BarExt};
use std::io::{stderr, IsTerminal, Result};

fn main() -> Result<()> {
    kdam::term::init(stderr().is_terminal());
    let mut pb = tqdm!(
        desc = "Training",
        colour = "orangered",
        dynamic_ncols = true,
        force_refresh = true,
        total = 300,
        unit = "steps",
        bar_format = "{desc suffix=''} | {percentage:.0}% = {count}/{total} {unit} | {rate:.1} {unit}/s | {remaining human=true} |{animation}|"
    );

    for _ in 0..300 {
        std::thread::sleep(std::time::Duration::from_secs_f32(0.025));
        pb.update(1)?;
    }

    pb.clear()?;
    pb.set_bar_format(
        "{desc suffix=''} | {total} {unit} | {rate:.1} {unit}/s | {elapsed human=true}",
    )
    .unwrap();
    pb.refresh()?;
    eprintln!();

    Ok(())
}

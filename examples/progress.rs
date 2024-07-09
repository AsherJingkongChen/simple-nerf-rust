extern crate kdam;

use kdam::{tqdm, BarExt};
use std::io::{stderr, IsTerminal, Result};

fn main() -> Result<()> {
    let epoch_count = 300;
    let mut bar = tqdm!(
        desc = "Training on 0 items",
        colour = "orangered",
        dynamic_ncols = true,
        force_refresh = true,
        total = epoch_count,
        unit = "steps",
        bar_format = "{desc suffix=''} {postfix} ┃ \
        {percentage:.0}% = {count}/{total} {unit} ┃ \
        {rate:.1} {unit}/s ┃ \
        {remaining human=true} \
        ┃{animation}┃"
    );
    bar.postfix = "┃ PSNR = 0.00 dB".into();

    kdam::term::init(stderr().is_terminal());
    for _ in 0..300 {
        std::thread::sleep(std::time::Duration::from_secs_f32(0.025));
        bar.update(1)?;
    }

    bar.clear()?;
    bar.set_bar_format(
        "{desc suffix=''} ┃ \
            {total} {unit} ┃ \
            {rate:.1} {unit}/s ┃ \
            {elapsed human=true}\n",
    )
    .unwrap();
    bar.set_description("Trained on 0 items");
    bar.refresh()?;

    Ok(())
}

# rizzmo

Code for my robot, Rizzmo.


## Hardware

TODO


## Software Setup

1. Install OS:
   1. Write the Jetson Nano Developer Kit SD card image to a microSD card ([instructions](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write)).
   2. Power up and install OS.
      - Hostname: `rizzmo`
   3. Remove unnecessary programs. Install updates.
2. Setup SSH:
   1. Copy `~/.ssh/id_rsa.pub` of dev machine to `~/.ssh/authorized_keys` on Jetson Nano.
   2. Set permissions: `chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys`
3. Clone this repo: `git clone https://github.com/austin-bowen/rizzmo.git`
4. Install and setup `pyenv` ([instructions](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)).
   - To update: `pyenv update`
5. Install Python 3.12: `pyenv install 3.12`
   - The latest version is installed.
6. Set Python 3.12 as default for `rizzmo` repo: `cd rizzmo && pyenv local 3.12`

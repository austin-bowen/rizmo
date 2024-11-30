# Rizmo

Code for my robot, Rizmo.

## Hardware

TODO

## Software Setup

1. Install OS:
    1. Write the Jetson Nano Developer Kit SD card image to a microSD
       card ([instructions](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write)).
    2. Power up and install OS.
        - Hostname: `rizmo`
    3. Remove unnecessary programs. Install updates.
2. Setup SSH:
    1. Copy `~/.ssh/id_rsa.pub` of dev machine to `~/.ssh/authorized_keys` on Jetson Nano.
    2. Set permissions: `chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys`
3. Install and setup `pyenv` ([instructions](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)).
    - To update: `pyenv update`
4. Install Python 3.12: `pyenv install 3.12 && pyenv global 3.12`
    - The latest version is installed and set to the default.
5. Set the default audio IO device:
   1. List devices:
      ```
      pactl list short sinks
      pactl list short sources
      ```
   2. Append lines to `/etc/pulse/default.pa`:
      ```
      set-default-sink <sink-name>
      set-default-source <source-name>
      ```

## Repository Setup

```bash
git clone https://github.com/austin-bowen/rizmo.git
cd rizmo
python -m venv --symlinks venv
. venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# For sounddevice lib
sudo apt install libportaudio2

# For serial communication
sudo usermod -aG dialout $USER
```

Setup git:

```bash
git config --global user.name "<name>"
git config --global user.email "<email>"
git config credential.helper store

# When trying to push, enter username, and paste personal access token
# generated from here: https://github.com/settings/tokens
```

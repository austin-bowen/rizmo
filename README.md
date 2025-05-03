# Rizmo

Code for my robot, Rizmo.

## Hardware

- Head:
  - [Wide-angle HD Webcam](https://a.co/d/b5ijvkz)
  - [Yahboom 2-DOF Servo Pan-Tilt Kit](https://a.co/d/96kztxG)
  - [180° Servo with Bracket](https://a.co/d/18uz2lH)
  - [Pololu Mini Maestro 12-Channel USB Servo Controller](https://a.co/d/fFQc6nA)
  - [DROK 12V to 5V 3A Buck Regulator](https://a.co/d/iQ9KW6e) for servo power
- Computer:
  - [NVIDIA Jetson Nano Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/product-development/)
  - [2x WiFi Antenna](https://a.co/d/2jopwcI)
  - [EMEET USB Conference Speaker](https://a.co/d/13vGQa1)
  - [DROK 8-32V to 3-27V 5A Buck Regulator](https://a.co/d/a1hP6Xy) for computer power
- Power:
  - [Zeee 7.4V 6200mAh 2S Lipo Battery](https://a.co/d/b4Y9Ils)
  - [8.4V 3A Li-ion Battery Charger](https://a.co/d/hgPVS8a)
  - [2S 18650 Lithium Battery Balance Charger Board](https://a.co/d/cqLRV7u)
  - [Lithium Battery Capacity Indicator Module - Green](https://a.co/d/2DSxyPZ)

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
4. Install Python 3.12 (with [optimizations](https://github.com/pyenv/pyenv/blob/master/plugins/python-build/README.md#building-for-maximum-performance)):\
   `env PYTHON_CONFIGURE_OPTS='--enable-optimizations --with-lto' PYTHON_CFLAGS='-march=native -mtune=native' pyenv install --verbose 3.12`
5. Set default Python version: `pyenv global 3.12`
6. Set the default audio IO device:
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
7. Install the [`jetson_inference`](https://github.com/dusty-nv/jetson-inference/tree/master) library:
   1. [Follow these instructions](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md) to install it from source to the system's Python 3.6 site packages.
   2. Create a Python 3.6 venv: \
      `python3.6 -m venv --system-site-packages --symlinks venv36`
   3. Update pip and make sure the lib can be imported:
      ```
      . venv36/bin/activate
      pip install -U pip
      pip install -r requirements.py36.txt
      python -c "import jetson_inference"
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

Create file `rizmo/secrets.py` with the following contents:

```python
AWS_ACCESS_KEY_ID: str = ...
AWS_SECRET_ACCESS_KEY: str = ...

OPENAI_API_KEY: str = ...

# Get from: https://developer.wolframalpha.com/access
# Make sure it has "Full Results API" permissions
WOLFRAM_ALPHA_APP_ID: str = ...
```

Make Rizmo start on system boot:

```bash
./bin/install-rizmo-service
sudo systemctl start rizmo
```

## Upgrading Python on Jetson Nano

```bash
# Install latest Python version
pyenv update
env PYTHON_CONFIGURE_OPTS='--enable-optimizations --with-lto' PYTHON_CFLAGS='-march=native -mtune=native' pyenv install --verbose 3.12

# Create new venv
mv venv venv.old
python -m venv --symlinks venv
. venv/bin/activate
python -V
# <verify it is using latest version>

# Install requirements
pip install -U pip
pip install -r requirements.txt
# <verify everything is working>

# Cleanup
rm -r venv.old
pyenv versions
pyenv uninstall <old-version>
```

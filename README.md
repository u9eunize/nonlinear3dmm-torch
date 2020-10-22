# NonLinear3DMM-torch

## TODO

- Docker automation


## Build

if you build fail

1. Install nvidia-container-runtime:

    `sudo apt-get install nvidia-container-runtime`

2. Edit/create the /etc/docker/daemon.json with content:
    ```
    {
        "runtimes": {
            "nvidia": {
                "path": "/usr/bin/nvidia-container-runtime",
                "runtimeArgs": []
             } 
        },
        "default-runtime": "nvidia" 
    }
    ```
3. Restart docker daemon:

    `sudo systemctl restart docker`

4. Build image (now GPU available during build):
    
# NonLinear3DMM-torch

## k8s settings

### nonlinear
```
apiVersion: v1
kind: Pod
metadata:
  name: ag-1-nonlinear-pretrain
  namespace: agtexture
  labels:
    app: ag-1-uvmap
spec:
  nodeName: g7-1
  volumes:
    - name: ag-pv-nonlinear-1
      persistentVolumeClaim:
        claimName: pvc-ag-1-nonlinear
    - name: dshm
      emptyDir:
        medium: Memory
  imagePullSecrets:
    - name: hpcd-registry-ag-registry
  containers:
    - name: ag-jhson
      image: "ag-registry.222.122.67.52.nip.io:443/nonlinear:proxy.1.0.11"
      command: ["/bin/bash"]
      args: 
        - "-c"
        - >-
          python setup.py install; 
          python train.py --config_json config/k8s/1-pretrain-albedo.json --checkpoint_regex "*/*";
          python train.py --config_json config/k8s/2-pretrain-shade-texture.json --checkpoint_regex "*/*";
          python train.py --config_json config/k8s/3-pretrain-comb.json --checkpoint_regex */*;
      resources:
        limits:
          cpu: 4000m
          memory: 14Gi
          nvidia.com/gpu: '1'
        requests:
          cpu: 4000m
          memory: 14Gi
          nvidia.com/gpu: '1'
      ports:
        - containerPort: 40000
      volumeMounts:
        - name: ag-pv-nonlinear-1
          mountPath: /data
        - name: dshm
          mountPath: /dev/shm
```

### tensorboard

```
apiVersion: v1
kind: Pod
metadata:
  name: ag-1-nonlinear-tensorboard
  namespace: agtexture
  labels:
    app: ag-1-uvmap
spec:
  nodeName: g7-1
  volumes:
    - name: ag-pv-nonlinear-1
      persistentVolumeClaim:
        claimName: pvc-ag-1-nonlinear
    - name: dshm
      emptyDir:
        medium: Memory
  imagePullSecrets:
    - name: hpcd-registry-ag-registry
  containers:
    - name: ag-jhson
      image: "ag-registry.222.122.67.52.nip.io:443/tensorboard:2.3.0"
      command: ["/bin/bash","-c"]
      args: 
        - "tensorboard --logdir /data/logs --host 0.0.0.0 --port 6006"
      resources:
        limits:
          cpu: 500m
          memory: 1Gi
        requests:
          cpu: 500m
          memory: 1Gi
      ports:
        - containerPort: 6006
      volumeMounts:
        - name: ag-pv-nonlinear-1
          mountPath: /data
        - name: dshm
          mountPath: /dev/shm
```

### using tensorboard

```
tensorboard --logdir_spec=./20201018,./20201019 --port 6006 --host 0.0.0.0
```

## Build

```
sudo docker rm nonlinear
sudo docker build -t ag-registry.222.122.67.52.nip.io:443/nonlinear:proxy.1.0.7 .
sudo docker run -it -v /data/project/nonlinear3dmm-torch:/data --gpus all --name nonlinear --ipc=host ag-registry.222.122.67.52.nip.io:443/nonlinear:proxy.1.0.6 /bin/bash -c "python setup.py install && python train.py --config_json config/exp-pretrain.json"
```

### Trouble Shooting
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
   
   
## TODO

- Docker automation


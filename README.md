# Voice transciption using Whisper AI (OpenAI)


### SETUP UP

#### Chocolatey - The Package Manager for Windows
- first need to be installed 
``` 
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

#### Pytorch
- second need to be installed
```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


#### Whisper AI

```
    pip install -U openai-whisper
```


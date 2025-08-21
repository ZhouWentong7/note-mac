

# 基本概念
Docker中三个重要的基本概念：镜像、容器、仓库。

## 镜像（Image）
 操作系统分为**内核**和**用户空间**。

Linux为例，用户所有的操作在自动挂在的root文件系统之下。Docker镜像就相当于一个root文件系统，只是包含了应用程序所需要的所有东西。

《Docker从入门到时间》中给出对镜像的定义：
> 是一个特殊的文件系统，除了提供容器运行时所需要的程序、库、资源和配置灯文件外，还包含了一些为运行时准备的一些配置参数（如匿名卷、环境变量、用户等）。镜像不包含任何动态数据，其内容在构建后也不会被改变。

特性：分层存储 
Docker中通过镜像来创建一个或多个容器。就像是程序里面的类和实例化的关系。

镜像是静态的，在容器中修改不会改变镜像的内容。

## 容器（Container）

本质是进程，但是运行于完全属于自己的命名空间，有自己的文件系统、网络配置、进程空间、用户ID空间等。

像自己在一个独立于宿主机的系统上操作，这种隔离带来了安全性的提升。

镜像是分层的，容器也是分层的。以镜像为基础层，其上构建一个自己的**容器存储层**（可读可写）。

容器是**易失的**（不能持久化存储数据）。想持久化存储数据，可以使用数据卷或者绑定宿主目录。


## 仓库

镜像构建完成后，可以方便地在宿主机上云霄，但若想在其他主机运行镜像，则需要**存储和分发**。**Docker Registry**就提供了这样的服务。

注册服务（Registry）将大量镜像分为多个仓库，每仓库有多个标签（Tag），每个标签对应一个镜像（Image)。

如一个Ubuntu镜像，仓库名为`ubuntu`，其标签通常是版本，如`22.04`，`24.04`。 那么可以使用`ubuntu:24.04`表示需要的版本的镜像。如果没有指定标签，则默认为最新版本的镜像。

**Docker Registry**支持多用户环境，完整仓库名多为`username/softwarename` ，类似`yyy/ubuntu:22.04`。

若不包含用户名，则指向官方镜像仓库。

# 基本命令

## `docker run`

从一个镜像创建一个容器并运行它：
```cmd
docker run <options> <image> <command>
```
- `<options>`：run的可用选项
- `<image>`：使用的镜像
- `<cmmand>`：创建容器后立刻执行的命令
	- 如果这个命令执行完后返回，则容器也会终止
	- 如果这个命令是一种挂起，如后端服务或者开启终端，则容器会一直运行。需要手动终止。

例子：
```
docker run alpine:latest echo "Hello world!"
```

先从本地寻找这个镜像，没有的话则会从官方查找并下载：
```
 (base) **➜**  **~** docker run alpine echo "hello world"          

Unable to find image 'alpine:latest' locally

latest: Pulling from library/alpine

6e174226ea69: Pull complete 

Digest: sha256:4bcff63911fcb4448bd4fdacec207030997caf25e9bea4045fa6c8c44de311d1

Status: Downloaded newer image for alpine:latest

hello world
```

1. 查找本地镜像，没有则下载到本地
2. 用准备好的镜像创建新的容器并启动
3. 执行command
4. 执行后**终止容器（并非删除）**

`docker run`常见命令选项:
- `-d/ --detach`:在后台运行容器并显示容器id
- `-e/ --env <variable=<value>`：设置环境变量
- `--rm`: 容器完成运行后删除容器
	- `docker run --rm <Image>`
- `--name <name>`：为创建的容器明明
- `-p/ --publish <host_port>:<container_port>`：容器中的端口映射到宿主机上
- `-it`：使用交互模式（开启标准输入）并分配未伪终端（可执行exit退出）

所以，命令可以变为：

```cmd
docker run --rm -it ubuntu
```

## `docker build`
构建镜像

常见选项:
- `-f/--file`，用于指定构建使用的Dockerfile
- `-t/--tag`：为镜像命名并添加tag

```
docker build -f path/to/Dockerfile -t test:1.0 .
```

后续会介绍如何写dockerfile

注意：命令最后有一个`.`。

## `docker images`
列出本地拥有的镜像。

## `docker ps`
列出正在运行的容器。

- `-a` 列出所有容器（包括没有运行的）

## 其他命令

| 命令示例 | 说明 |
|----------|------|
| `docker exec <options> <container> <command>` | 在一个正在运行的容器中执行命令 |
| `docker attach <options> <container>` | 连接到后台运行的容器，按下 `CTRL+P CTRL+Q` 可以断开连接 |
| `docker start <options> <containers>` | 启动一个或多个已停止运行的容器 |
| `docker stop <options> <containers>` | 停止一个或多个正在运行的容器 |
| `docker rm <options> <containers>` | 删除一个或多个容器 |
| `docker rmi <options> <images>` | 删除一个或多个镜像 |
| `docker cp <options> <container>:<src_path> <dest_path>` 或 `docker cp <options> <src_path> <container>:<dest_path>` | 在宿主机和容器间复制文件/文件夹 |

### 一个使用流程案例
```cmd
(base) **➜**  **~** docker run -it --rm -d ubuntu         

Unable to find image 'ubuntu:latest' locally

latest: Pulling from library/ubuntu

49a8ca9a328e: Pull complete 

Digest: sha256:7c06e91f61fa88c08cc74f7e1b7c69ae24910d745357e0dfe1d2c0322aaf20f9

Status: Downloaded newer image for ubuntu:latest

afa04e1f18cfcc6f89003ac3ed9d962393806a66ef4b8bc1a7455f91b36e978a

(base) **➜**  **~** docker ps                    

CONTAINER ID   IMAGE     COMMAND       CREATED         STATUS         PORTS     NAMES

afa04e1f18cf   ubuntu    "/bin/bash"   8 seconds ago   Up 8 seconds             laughing_clarke

(base) **➜**  **~** docker attach afa04e1f18cf

root@afa04e1f18cf:/#
```
使用attach就是将这个容器挂在前台

然后使用ctl+P和ctrol+Q连续按退出容器，但没有停止运行。

使用`docker stop <container_id>`停止。

## 删除镜像

要删除一个镜像，首先需要全部保证其容器全部暂停
# 使用Dockerfile定制镜像

## 基本语法

Dockerfile是一种脚本文件。

### `FROM <image>`
指定基础镜像，我们定制的镜像在基础镜像之上构建
```
FROM ubuntu:22.04
```

### `RUN <command>`
在镜像中执行命令，是常用指令。

```cmd
RUN echo -e "hello." > text.txt
```

此外，exec格式的写法为：

```cmd
RUN ["可执行文件"， "参数1"， "参数2"]
```

```
RUN ["apt", "install", "vim"]
```

### `WORKDIR <dir_path>`
指定工作目录(当前目录，相对路径就是相对于当前目录的)，不存在则会自动创建

```cmd
WORKDIR /app
```

### `COPY <src_paths> <dest_path>`

用于复制文件，将构建上下文目录中`<src_paths>`的文件/目录，复制到新一层镜像的`<dest_path>`位置
- 在构建镜像时，把 **宿主机上的文件或目录** 复制到 **镜像中的指定路径**。
- 常用于将应用代码、配置文件、脚本等放进镜像里。

- `<src>`：源文件或目录路径（相对 Dockerfile 所在目录，也可以是构建上下文里的路径）。
 - `<dest>`：容器内的目标路径（通常写绝对路径，如 /app/）。

```cmd
# 把当前目录下的所有文件复制到容器的 /app 目录
COPY . /app

# 复制单个文件
COPY requirements.txt /app/requirements.txt

# 复制多个文件到同一个目标目录
COPY main.py config.yaml /app/
```

也可以写作
``` 
COPY ["main.py","config.yaml","/app/"]
```

### `ENV <key1>=<value1> <key2>=<value2>`
### `EXPOSE <ports>`

**声明**容器暴露的端口，帮助使用者理解镜像打算用什么端口，方便配置端口映射。**只是声明，并非自动进行端口映射。**

### `CMD <command>`
作为容器启动命令，即创建容器并自动立即执行的命令，可以被`docker run <image> <command>`中提供的命令替换。

```cmd
CMD ["可执行文件"， "参数1"， "参数2"]
```


## 完整Dockerfile
在文件下创建Dockerfile：
>GPT 生成的DL环境，适用与Win和Mac

```dockerfile
# 使用官方 Python 3.11 slim 版本，适配 x86_64 和 ARM64
FROM python:3.11-slim

# 设置工作目录
WORKDIR /workspace

# 避免 Python 输出缓冲
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl vim build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装通用 PyTorch（CPU 版本），兼容 Windows/Linux/Mac
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    numpy pandas matplotlib seaborn scikit-learn jupyterlab tqdm opencv-python

# 设置 PyTorch MPS（Mac GPU）环境变量
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# 可选：复制项目代码
# COPY . /workspace

# 暴露 Jupyter Lab 端口
EXPOSE 8888

# 默认启动 Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

创建镜像：

```cmd
docker build -t dl-env:1.0 .
```

构建镜像：
```
docker run --gpus all -it -p 8888:8888 -v $(pwd):/workspace dl-env
```
启动一个基于 dl-env 镜像的容器，启用 GPU，进入交互模式，把宿主机当前目录挂载到容器 /workspace，并把容器里的 **JupyterLab (8888)** 映射到宿主机的 **8888 端口**。

如果不使用GPU的话：

```
 
```
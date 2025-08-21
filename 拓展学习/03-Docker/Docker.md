
# 基本概念
Docker中三个重要的基本概念：镜像、容器、仓库。

## 镜像（Image）
 操作系统分为**内核**和**用户空间**。

Linux为例，用户所有的操作在自动挂在的root文件系统之下。Docker镜像就相当于一个root文件系统，只是包含了应用程序所需要的所有东西。

《Docker从入门到实践》中给出对镜像的定义：
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

### 简单的CPP环境

在文件夹下创建一个cpp的main文件，写一个简单的输出：

```cpp
#include<iostream>

using namespace std;

  

int main() {

	cout << "Hello, CPP docker!" << endl;

	return 0;

}
```

dockerfile:

```txt
FROM ubuntu

RUN apt update

ENV DEBIAN_FRONTEND=noninteractive

RUN apt install -y build-essential

WORKDIR /usr/src/cpp

# 将main.cpp从宿主机复制到上述文件夹下
COPY main.cpp .

RUN g++ -o main main.cpp

CMD ["./main"]

```

创建：

```
(base) ➜  cpp1 docker build -t cpp-env:1.0 .
```

执行：

```
(base) ➜  cpp1 docker run --rm cpp-env:1.0    
```               
> Hello, CPP docker!

### 如果需要多个文件编译，可以使用makefile等

### 构建简单的DL环境
在文件下创建Dockerfile：
>GPT 生成的DL环境，适用与Win和Mac

```dockerfile
FROM condaforge/miniforge3

WORKDIR /workspace


# 避免 Python 输出缓冲
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl vim build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装通用 PyTorch（CPU 版本），兼容 Windows/Linux/Mac
# 安装 Python + PyTorch
RUN conda install -y python=3.11 \
    pytorch torchvision torchaudio -c pytorch -c conda-forge \
    && conda clean -a -y

# 安装其他数据科学库
RUN conda install -y numpy pandas matplotlib seaborn scikit-learn jupyterlab tqdm opencv \
    -c conda-forge \
    && conda clean -a -y
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
dl-env docker run -it --rm -p 8888:8888 -v $(pwd):/workspace dl-env:1.0 
```
启动一个基于 dl-env 镜像的容器，启用 GPU，进入交互模式，把宿主机当前目录挂载到容器 /workspace，并把容器里的 **JupyterLab (8888)** 映射到宿主机的 **8888 端口**。

- --rm 参数的作用是 **容器停止后自动删除容器**。
- 也就是说，你关闭 Jupyter Lab 或按 Ctrl+C 停止容器后，Docker 会自动清理这个容器。 
- 优点：不会占用磁盘空间。
- 缺点：如果你在容器里修改了数据，但没有挂载本地目录，修改的数据会丢失。

根据控制台的输出，可以进入jupyter notebook的页面。
## 镜像是如何构建的

镜像具有**分层存储**的特性，而Dockerfile的**每一行命令**都会建立一个新的层。每一层都在上一层基础上机械能修改，每层构建后不会再发生改变，任何改变都只发生在自己的层，而不会影响前面的层。

假设有个Dockerfile：
```
FROM python:3.11-slim      # Layer 1: 基础镜像
RUN apt-get update && apt-get install -y git wget  # Layer 2: 系统依赖
COPY requirements.txt /workspace/                # Layer 3: 拷贝文件
RUN pip install -r /workspace/requirements.txt  # Layer 4: Python依赖
```

- **Layer 1**：Python 基础镜像，所有使用 Python 镜像的 Dockerfile 都可以复用。
- **Layer 2**：安装 git/wget，只在这个 Dockerfile 中生成。
- **Layer 3**：拷贝 requirements.txt 文件，只在这一步修改时重新构建。
- **Layer 4**：安装 Python 库，基于 Layer 3 增量构建。  

**好处：**

- 如果你修改了 requirements.txt，只有 Layer 3 和 Layer 4 会重新构建，Layer 1 和 Layer 2 可以复用缓存。     
- 生成的镜像体积更小，因为共享了重复的层。


## 构建镜像上下文

`docker build`命令的最后那个`.`。

文件系统中`.`表示当前目录，但是在Dockerfile中，路径是由`-f`指定的，这个`.`值得是上下文路径。
> 一般来说，dockerfile都会在项目的根目录下，所以`.`是最常用的。


在这个路径显得文件会被上传用于构建镜像，所以 Docker **只能访问上下文里的文件**，上下文外的文件无法被 COPY 或 ADD。

例子：

```
FROM python:3.11-slim
COPY requirements.txt /workspace/
RUN pip install -r /workspace/requirements.txt
COPY . /workspace/
```

- `COPY requirements.txt /workspace/`
    → 只能复制 **构建上下文内** 的 requirements.txt。如果你写了 `../requirements.txt`，Docker 会报错。
- `COPY . /workspace/`
    → 把上下文里的所有文件都复制到容器 `/workspace`。

也可以看出，如果上下文很大，则构建速度会很慢，可以写`.dockerignore`文件排除不需要的文件夹和文件，例如：

```
data/
*.log
.git/
```

## 更实际的使用场景Djiango

略

## 多阶段构建

之前都是一次性完成所有的构建，最终得到的镜像里面除了可执行文件之外，还有原始项目文件、编译库等，但其实，以C++为例，编译后无需再利用原始代码和库，可以在任意ubuntu下运行，这些只会增加镜像大小。

所以使用多阶段构建，将构建过程分成多个阶段完成，每一个FROM都是一个阶段的开始：

```
# build

FROM ubuntu AS builder

RUN apt update

ENV DEBIAN_FRONTEND=noninteractive

RUN apt install -y build-essential

WORKDIR /usr/src/cpp

COPY main.cpp .

RUN g++ -o main main.cpp

# runtime
FROM ubuntu AS runtime

COPY --from=builder /usr/src/cpp/main .

CMD ["./main"]

```

可以看到image的大小明显缩水：

```

base) ➜  cpp2 docker images
REPOSITORY                 TAG       IMAGE ID       CREATED             SIZE
cpp-env                    2.0       9e41e2c5e16a   2 minutes ago       101MB
dl-env                     1.0       b971fb0aaaec   About an hour ago   8.18GB
cpp-env                    1.0       7d22325bcd01   About an hour ago   465MB
```

- builder 阶段用来做 “重的编译工作”，不影响最终镜像大小



第二阶段也不一定非要使用ubuntu，可以换位scrach这种纯空的镜像，在里面安装cpp使用的库。

# 数据管理

持久化数据需求。

后端代码更新后，需要重新构建镜像并创建新的容器，则原来的容器中的数据就消失了。

可以使用Docker的数据卷和挂在主机目录两种方式来避免这个问题。

## 数据卷

一个可供一个或多个容器使用的特殊目录，可以绕过Docker的联合文件系统：
- 容器间共享
- 对其的修改立刻生效
- 对数据卷的更新不影响镜像
- 数据卷默认一直存在，即使容器被删除

### 创建数据卷

```
docker volume create <vol_name>
```

### 查看数据卷

```
docker volume ls
```

### 查看所有数据卷信息

```
docker volume inspect <vol_name>
```

### 挂在数据卷

在docker run的时候挂在数据卷


```
docker run --rm --mount source=<vol_name>,target= path/to/data
```


例子：
```
docker run -it --rm \
    --mount source=mydata,target=/workspace/data \
    dl-env:1.0 /bin/bash
```

> 注意
> 1. **不要留空格**
>    ```
> --mount source=mydata,target=/workspace/data   # 正确
--mount source=mydata, target=/workspace/data  # ❌ 空格会报错
>    ```
> 2. --mount 是 **推荐的新语法**，比 -v mydata:/workspace/data 更明确，尤其是生产环境。
> 3. target **必须是容器内路径**，不能是宿主机路径。
### 删除卷
```
docker volume rm mydata
```
## 挂载主机目录

挂载主机目录必须使用**绝对路径**，默认权限为**读写**。
```
docker run -it --rm \
    --mount type=bind,source=/path/on/host,target=/workspace \
    dl-env:1.0 /bin/bash
```

- type=bind → 表示挂载主机目录
- source → 宿主机目录
- target → 容器内路径

可以增加`readonly`选项控制**只读**。
```
docker run --name web_backend \
	--mount type=bind,source=/usr/apps/web_database,target=/app/database,readonly \
	backend
```

也可挂载主机文件：

```
docker run --name web_backend \
	--mount type=bind,source=/usr/apps/web_app/.env,target=/app/.env,readonly \
	backend 
```

可以多个挂载：
略（多写几行--mount)

# Docker Compose
一个大项目往往需要多个应用组成，需要输入多个Docker命令，且数据卷挂载和容器间通信更为糟糕。

Compose就是一个**容器集群快速编排**的工具。

使用YAML文件配置项目容器集群，并通过`docker compose`快速启动

## 配置一组容器
`Docker-Compose.yml`选项与Docker命令基本一致，只是需要改为YAML写法。 

下面是GPT给的一个前后端案例：

假设你有一个 Django Web 项目 + PostgreSQL 数据库。

我们用 Docker Compose 管理 **两个服务**：

1. web → Django 后端
    
2. db → PostgreSQL 数据库
    

---
docker-compose.yml 示例

```
version: "3.9"

services:
  web:
    image: my_django_app:latest   # 已构建的 Django 镜像
    container_name: web_backend
    ports:
      - "8000:8000"               # 映射宿主机端口
    volumes:
      - ./web_app:/app             # 挂载本地项目目录
      - ./web_app/.env:/app/.env:ro  # 挂载只读环境文件
    depends_on:
      - db                        # 确保 db 服务先启动
    environment:
      - DEBUG=1

  db:
    image: postgres:15
    container_name: postgres_db
    restart: always
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:
```

---

对应 Docker 命令解释

| **Compose 配置**                            | **对应的 Docker 命令**                                                                |
| ----------------------------------------- | -------------------------------------------------------------------------------- |
| image: my_django_app:latest               | `docker run my_django_app:latest`                                                |
| ports: - "8000:8000"                      | `docker run -p 8000:8000`                                                        |
| volumes: - ./web_app:/app                 | `docker run -v $(pwd)/web_app:/app`                                              |
| volumes: - ./web_app/.env:/app/.env:ro    | `docker run -v $(pwd)/web_app/.env:/app/.env:ro`                                 |
| depends_on: - db                          | 手动顺序启动 `docker run -d db `再 `docker run web`                                     |
| environment:                              | `docker run -e DEBUG=1`                                                          |
| volumes: db_data:/var/lib/postgresql/data | `docker volume create db_data && docker run -v db_data:/var/lib/postgresql/data` |

---

使用 Docker Compose 命令

1. 构建镜像（如果 Dockerfile 在当前目录）：
    

```
docker compose build
```

2. 启动所有服务：
    

```
docker compose up
```

- 使用 -d 后台运行：
    

```
docker compose up -d
```

3. 停止服务：
    

```
docker compose down
```

- 会停止并删除容器，但默认不会删除卷（数据持久化保留）。
    

  

4. 查看日志：
    

```
docker compose logs -f web
```

---

 优点

- 不用手动写多个 docker run 命令，Compose 自动管理依赖顺序和网络。
    
- 可以挂载卷、设置环境变量、映射端口，非常适合开发和多服务部署。
    

---

1. [Docker官方](https://docs.docker.com/get-started/introduction/build-and-push-first-image/)
2. [Docker从入门到实践](https://yeasy.gitbook.io/docker_practice)

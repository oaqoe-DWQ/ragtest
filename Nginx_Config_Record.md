# Nginx 反向代理配置记录

## 需求

- 本地访问：`http://172.21.230.235:8100`（保留）
- 域名访问：`http://QA.pg:8100`（对外暴露域名，隐藏 IP）
- 别人访问：`http://QA.pg`（通过域名访问服务）

## 配置步骤

### 1. 下载并安装 Nginx

从 [nginx.org](https://nginx.org/en/download.html) 下载 Windows 版本。

```powershell
# 下载
Invoke-WebRequest -Uri "https://nginx.org/download/nginx-1.25.4.zip" -OutFile "$env:TEMP\nginx.zip"

# 解压到 C:\nginx
Expand-Archive -Path "$env:TEMP\nginx.zip" -DestinationPath "C:\nginx" -Force
```

### 2. 创建 Nginx 配置文件

创建 `C:\nginx\nginx-1.25.4\conf\conf.d\qa.conf`：

```nginx
server {
    listen 80;
    server_name QA.pg;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8100;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support if needed
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 3. 修改主配置文件

编辑 `C:\nginx\nginx-1.25.4\conf\nginx.conf`，在 `http {}` 块中添加：

```nginx
include conf.d/*.conf;
```

### 4. 配置 hosts 文件

以管理员权限运行 PowerShell，添加 hosts 记录：

```powershell
Add-Content -Path "C:\Windows\System32\drivers\etc\hosts" -Value "172.21.230.235    QA.pg"
```

### 5. 启动 Nginx

```powershell
# 启动
C:\nginx\nginx-1.25.4\nginx.exe

# 测试配置
C:\nginx\nginx-1.25.4\nginx.exe -t

# 重新加载配置
C:\nginx\nginx-1.25.4\nginx.exe -s reload

# 停止
C:\nginx\nginx-1.25.4\nginx.exe -s stop
```

### 6. 验证配置

```powershell
# 检查端口 80 是否监听
netstat -ano | findstr ":80"

# 测试访问
Invoke-WebRequest -Uri "http://QA.pg" -UseBasicParsing
```

## 常用命令

| 操作 | 命令 |
|------|------|
| 启动 | `nginx.exe` |
| 停止 | `nginx.exe -s stop` |
| 重载配置 | `nginx.exe -s reload` |
| 测试配置 | `nginx.exe -t` |
| 查看进程 | `Get-Process nginx` |
| 强制结束 | `Get-Process nginx \| Stop-Process` |

## 文件路径

| 文件 | 路径 |
|------|------|
| Nginx 主程序 | `C:\nginx\nginx-1.25.4\nginx.exe` |
| 主配置文件 | `C:\nginx\nginx-1.25.4\conf\nginx.conf` |
| 站点配置 | `C:\nginx\nginx-1.25.4\conf\conf.d\qa.conf` |
| hosts 文件 | `C:\Windows\System32\drivers\etc\hosts` |

## 别人如何访问

需要在访问者的电脑上配置 hosts 文件：

```
172.21.230.235    QA.pg
```

或者联系公司/学校的 DNS 管理员，在内网 DNS 服务器上添加 `QA.pg` 的 A 记录指向 `172.21.230.235`。

## 开机自启（可选）

### 方法 1：启动文件夹

创建快捷方式到 `C:\nginx\nginx-1.25.4\nginx.exe`，放入：
```
shell:startup
```

### 方法 2：任务计划程序

使用任务计划程序设置开机启动。

### 方法 3：NSSM 注册为服务

```powershell
# 安装 NSSM
winget install nssm

# 注册服务
nssm install nginx "C:\nginx\nginx-1.25.4\nginx.exe" "-c C:\nginx\nginx-1.25.4\conf\nginx.conf"
```

## 注意事项

1. 如果 80 端口被占用，需要先停止占用 80 端口的服务
2. 修改配置后需要执行 `nginx -s reload` 重新加载
3. Windows 防火墙可能需要允许 nginx.exe 通过
4. 确保后端服务（8100 端口）正常运行

## 端口占用检查

```powershell
# 查看 80 端口占用
netstat -ano | findstr ":80"

# 查看具体进程
tasklist /fi "pid eq <PID>"
```

# Outlook 个人邮箱发信配置指南

Outlook 个人邮箱（`@outlook.com` / `@hotmail.com`）**已禁用 SMTP 密码登录**，必须使用 **你自己注册的 Azure 应用** + **Microsoft Graph OAuth**。

> ⚠️ 不能使用 Microsoft 官方公共客户端 ID（如 Azure PowerShell），否则会报错：
> `first party application... users are not permitted to consent`

## 一、注册 Azure 应用（约 5 分钟）

1. 打开 [Azure 应用注册](https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps/ApplicationsListBlade)
   - 没有 Azure 账号？用 `lixiaohui2020@outlook.com` 免费注册即可

2. 点击 **「新注册」**
   - **名称**：`Paper Digest`（随意）
   - **支持的账户类型**：选 **「个人 Microsoft 账户」**
   - **重定向 URI**：平台选 **「公共客户端/本机 (移动设备和桌面)」**，URI 填 `http://localhost`
   - 点击 **注册**

3. 复制 **应用程序(客户端) ID**（格式如 `a1b2c3d4-e5f6-7890-abcd-ef1234567890`）

4. 左侧 **「API 权限」** → **添加权限** → **Microsoft Graph** → **委托权限** → 勾选 **`Mail.Send`** → 添加

5. 左侧 **「身份验证」** → 拉到最下 **「高级设置」** → **允许公共客户端流** → 选 **是** → 保存

## 二、写入配置

编辑 `configs/paper_digest.yaml`：

```yaml
email:
  auth_method: graph
  recipient: "lixiaohui@allwinnertech.com"
  smtp_user: "lixiaohui2020@outlook.com"
  graph:
    client_id: "你的应用程序客户端ID"
    authority: "https://login.microsoftonline.com/consumers"
```

或设置环境变量：

```bash
export PAPER_DIGEST_GRAPH_CLIENT_ID="你的应用程序客户端ID"
```

## 三、一次性 OAuth 授权

```bash
pip install -e ".[paper-digest]"
python -m scripts.paper_digest.setup_outlook_oauth
```

终端会显示设备码，打开 https://microsoft.com/link 输入，用 `lixiaohui2020@outlook.com` 登录并同意权限。

## 四、测试发送 & 定时任务

```bash
python -m scripts.paper_digest.main          # 发送测试
bash scripts/install_paper_digest_cron.sh    # 每天 8:00
```

## 替代方案

若不想注册 Azure 应用，可改用支持 SMTP 应用密码的邮箱发信：

| 邮箱 | auth_method | SMTP |
|------|-------------|------|
| Gmail | `smtp` | smtp.gmail.com:587 |
| QQ 邮箱 | `smtp` | smtp.qq.com:587 |
| 163 邮箱 | `smtp` | smtp.163.com:465 |

发件人可改为上述邮箱，收件人仍为 `lixiaohui@allwinnertech.com`。

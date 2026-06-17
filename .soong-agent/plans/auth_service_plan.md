# 用户身份验证服务模块添加计划 (Auth Service Implementation Plan)

## 1. Goal
在现有项目中成功集成一个用户身份验证服务，实现用户注册、基于用户名/密码的登录流程，并提供获取当前用户信息的功能，确保系统具备安全、可扩展的用户管理能力。

## 2. Scope
*   **核心模块:** 创建 `auth` 服务层和相应的 API 网关接口。
*   **功能点:**
    *   用户注册 (`/api/v1/auth/register`)：接收用户名、密码，创建加密后的用户记录。
    *   用户登录 (`/api/v1/auth/login`)：验证提供的凭证，并返回一个包含有效期的 JWT Token。
    *   获取用户信息 (`/api/v1/user/me`)：使用传入的 Token 验证身份，并返回对应的用户信息（如ID、用户名）。
*   **数据存储:** 使用现有数据库连接层，增加 `users` 表或相应的用户模型。

## 3. Approach
**选择策略 (Chosen Strategy):**
采用基于 JWT (JSON Web Token) 的无状态认证机制。服务将负责处理密码哈希（使用 Bcrypt 或 Argon2）和 Token 生成/验证。API 网关层应在所有需要身份验证的路由前进行拦截和校验。

**拒绝替代方案 (Rejected Alternatives):**
1.  **Session-based Auth:** 拒绝使用传统的服务器端 Session，因为它会增加服务状态管理（State Management）的复杂性，降低可扩展性和横向伸缩能力。
2.  **OAuth/OIDC 集成:** 暂不集成第三方 OAuth 提供商（如 Google/GitHub），本次迭代仅聚焦于核心的用户名/密码认证流程，以最小化范围和快速交付价值。

## 4. Interfaces (变更接口)
*   **数据库层:**
    *   新增 `users` 表：包含 `id` (UUID), `username` (VARCHAR, UNIQUE), `password_hash` (VARCHAR), `created_at` (TIMESTAMP)。
    *   需要一个密码哈希函数（如 `hash_password(plaintext)`）和验证函数（如 `verify_password(hash, plaintext)`）。
*   **API 接口:**
    *   `POST /api/v1/auth/register`: Body: `{ "username": string, "password": string }`. Response: `{ success: boolean, message: string }`.
    *   `POST /api/v1/auth/login`: Body: `{ "username": string, "password": string }`. Response: `{ token: string, expires_at: datetime }`.
    *   `GET /api/v1/user/me`: Header: `Authorization: Bearer <token>`. Response: `{ id: UUID, username: string, email: string }`.
*   **内部工具:** 必须新增一个 Token 生成和解析的内部服务。

## 5. Steps (实施步骤)
1.  **[Database]** 在数据库迁移脚本中创建或更新 `users` 表结构，并确保密码哈希字段的索引和约束。
2.  **[Core Logic]** 实现用户模型层：添加注册逻辑（包括密码哈希）和凭证验证逻辑。
3.  **[Service Layer]** 创建 `AuthService` 类/模块，封装业务流程：调用数据库进行查找 -> 验证密码 -> 生成 JWT Token。
4.  **[API Gateway]** 在 API 网关层实现认证中间件（Middleware）：拦截所有受保护的路由，从 Header 中提取 Token，并调用 `AuthService` 进行校验。
5.  **[Endpoint Implementation]** 实现 `/register`, `/login`, 和 `/user/me` 的控制器和请求处理逻辑。

## 6. Edge Cases (边缘情况与风险)
*   **密码安全:** 必须使用至少 Bcrypt 或 Argon2 等高强度、自适应的哈希算法，并确保盐值（Salt）是随机生成的。
*   **并发注册:** 需要处理用户名已存在时的唯一性约束和冲突错误。
*   **Token 过期/吊销:** 登录失败或用户主动登出时，需要考虑 Token 的过期机制和潜在的黑名单/吊销列表管理。
*   **输入校验:** 所有输入（用户名、密码）必须进行严格的长度和字符集校验，防止注入攻击。

## 7. Verification (验证方法)
1.  **单元测试:** 为 `AuthService` 的注册、登录、Token生成等核心逻辑编写覆盖率达到 90% 以上的单元测试。
2.  **集成测试:** 使用 Postman 或类似工具，执行完整的用户生命周期流程：
    *   Test Case 1: 新用户成功注册 -> Test Case 2: 使用新账号成功登录并获取 Token -> Test Case 3: 使用过期/无效 Token 访问受保护资源（应返回 401）。
3.  **安全审计:** 对所有输入点进行 XSS 和 SQL 注入的渗透测试。

## 8. Assumptions (假设)
*   项目已配置了标准的数据库连接池和 ORM 层，可供本模块使用。
*   JWT 的密钥（Secret Key）将在配置文件中提供，且该密钥必须是高熵值、保密的。
*   系统时间同步准确，用于 Token 的过期时间计算。
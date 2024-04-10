## Workload identity federation

---

### 课题概述

通过Github的Actions功能，我希望能自动上传merge好的main分支的内容到GCP的对应bucket中，实现代码deploy的自动化。为了不使用SA也能安全操作GCP中的资源，使用到了这个服务。但一知半解让我很难受，所以进行了查找和总结，我觉得我还是没有完全理解所有的细节，但是从整体来说我有了理解。

### workload identity的工作流程和设置内容

我们在使用一个application去访问GCP云环境的时候，一般来说需要用到service account（SA），通过SA的密钥，进入GCP作业。但是你无法保证这个密钥的安全性，它们在外部的应用手中，没有使用期限，更没有安全保证。而Workload identity federation则提供了解决方案。

假设有如下parts：外部应用有identity provider，同时有需要执行的workloads。云内部则是SA和需要通过SA执行操作的resource服务。如何把这两者联系起来。

设置之前的流程：

- 身份验证（Authentication）：Workload（工作负载）首先从自己的应用程序中的身份提供商（Identity Provider）进行身份验证。这意味着 Workload 向身份提供商发送身份验证请求，并提供用户的凭据（例如用户名和密码）。身份提供商验证用户的凭据，并返回一个带有用户身份信息的身份令牌（ID Token）或认证凭据（Account Credentials）。
- 授权（Authorization）：接下来，Workload 使用从身份提供商获取的认证凭据，向 Google Cloud Platform（GCP）的安全令牌服务（Security Token Service）请求一个短暂的访问令牌（Access Token）。这个访问令牌允许 Workload 代表特定的服务账号（Service Account）执行对资源的操作。
- 然后workload就可以使用这个token披上SA的外衣，进行对resource的操作了。

设置流程：（code例子）

- 在GCP的项目中创建workload-identity-pools。
```bash
gcloud iam workload-identity-pools create pool-id \
    --location='yourlocation' \
    --description='description' \
    --display-name='display-name'
```
进行这个设置，只要对SA，IAM和workload-identity对访问权限即可。

- Workloadidentity可以设置很多pools对每个外部ID都进行授权和设置。在workloadidentity中可以设置关于外部id提供商的各种内部信息，通过属性进行设置。
```bash
gcloud iam workload-identity-pools providers create-oidc provider-id \
    --workload-identity-pool='pool-id' \
    --issuer-uri='issuer-uri' \
    --location='yourlocation' \
    --attribute-mapping='google.subject=assertion.sub'
```

- 需要对SA设置使用权限，这个设置可以通过命令行，也可以在GUI执行，执行的时候要去SA的page，然后点击权限，加入针对SA的权限。用户的iam principal可以在workloadidentity的画面确认到，就是下面的member位置的样子，看起来完全不是一个用户的样子，但是确实是对这个东西授权了对SA对使用权限。（注意这个设置每个人的都不一样，请查过再设置）

```bash
gcloud iam service-accounts add-iam-policy-binding service-account-email \
    --role roles/iam.workloadIdentityUser \
    --member 'principal://iam.googleapis.com/projects/project-number/locations \
                /global/workloadIdentityPools/pool-id/subject/subject'
```

- 对你的SA设置相应的工作权限，比如我要对对应的GCS桶执行文件上传操作，那么就需要对SA设置相应的IAMrole权限。

通过以上操作，这时候设置后的流程就变成了：在Workload通过credentials，去到GCP的security token service要求token的时候，security token service就会去workload identity pool中验证请求者的身份是否合规。

是的，您可以像原来那样，使用 STS 发行的访问令牌代替服务账号进行操作。Workload Identity 的主要作用是提供了一种更安全、更可管理的方式来将外部身份与服务账号关联起来。下面是使用 Workload Identity 相对于直接使用 STS 发行的访问令牌的一些优势和区别：

1. **安全性**：Workload Identity 提供了一种更加安全的认证机制。通过与外部身份提供者（IdP）进行集成，工作负载可以利用现有的身份验证机制，如 OpenID Connect（OIDC），进行身份验证。这样可以避免在代码中硬编码密钥或凭据，降低了密钥管理的风险。

2. **可管理性**：使用 Workload Identity 可以更好地管理服务账号的访问权限。通过为工作负载配置适当的 Workload Identity Pool 和 Provider，可以精确控制哪些工作负载可以代表哪些服务账号执行操作，以及可以执行哪些操作。

3. **审计追踪**：Workload Identity 可以提供更好的审计追踪功能。通过将外部身份与服务账号关联起来，可以更容易地跟踪和记录谁在何时执行了哪些操作，从而提高了审计的可追溯性和透明度。

总的来说，虽然您可以直接使用 STS 发行的访问令牌代替服务账号进行操作，但是使用 Workload Identity 可以提供更安全、更可管理的身份验证和授权机制，使您的应用程序更加健壮和可靠。

关于设置，可以通过上述的命令也可以通过GUI画面进行设置，设置时候的详细到具体内容的东西不再概述。

只举出要点：

- 每个OIDC提供者都有自己的服务属性比如Github你可以设置如下属性mapping：(见第二行，定义了workload的工作repo)

```
google.subject = assertion.sub 
attritube.repository = assertion.repository
```
- 在属性条件中通过CEL格式设置只有在该条件下才能执行认证的设置。比如如下方式，设置只有在myrepo中发生的工作流才能授予权限。

```
assertion.repository == 'xxx/myrepo'
```
### 补充知识

1，OIDC

OIDC（OpenID Connect）是一种身份验证协议，它建立在 OAuth 2.0 协议之上，提供了身份验证（Authentication）和用户信息（UserInfo）获取的功能。OIDC 旨在通过在客户端和认证服务器之间建立信任关系来实现用户身份验证。

在上面的验证流程中，OIDC 的作用体现在身份验证阶段，其中 Workload 与身份提供商进行交互以验证用户的身份，并获取与用户相关的凭据。OIDC 提供了一种标准化的身份验证流程，使得 Workload 能够安全地从自己的应用程序中验证用户，并且能够与其他身份提供商进行集成，如 GCP 的安全令牌服务。通过 OIDC，Workload 能够在身份验证成功后获取访问 GCP 资源所需的访问令牌，从而实现对资源的授权访问。

2，IdP

IdP 指的是身份提供者（Identity Provider），是指能够验证用户身份并颁发身份令牌（如 ID Token）或认证凭据（这里用到了）的服务。在身份验证过程中，用户向身份提供者提供凭据（例如用户名和密码、多因素认证信息等），身份提供者验证这些凭据的有效性，并确定用户的身份是否可信。如果验证成功，身份提供者会向用户颁发一个身份令牌，其中包含有关用户身份的信息，供其他服务进行使用。

身份提供者可以是多种形式的身份验证服务，包括但不限于：

- 组织内部的身份验证服务：由组织自行搭建和管理的身份验证系统，用于验证组织内部员工、合作伙伴或客户的身份。
- 第三方身份验证服务：由第三方提供的身份验证服务，例如 Google、Facebook、GitHub 等，用户可以使用其账号进行登录和身份验证。
- 单点登录（SSO）提供者：专门用于提供单点登录功能的服务，允许用户在登录后访问多个相关联的应用程序，而无需多次进行身份验证。

身份提供者在身份验证过程中扮演着核心的角色，负责验证用户身份并颁发相应的凭据，以便用户可以安全地访问各种服务和资源。

- 3，STS

STS 指的是安全令牌服务（Security Token Service），是一种用于颁发临时安全凭证的服务。在身份验证和授权过程中，STS 可以颁发临时访问令牌（如上面说的云服务的的临时安全凭证token）给客户端，以便客户端可以安全地访问受保护的资源。

STS 的主要功能包括：
- 颁发临时凭证：STS 可以颁发临时的安全凭证，例如临时访问令牌、身份令牌等，这些临时凭证具有一定的有效期，在有效期内可用于访问特定的资源。
- 角色扮演（Role assumption）：STS 可以支持角色扮演机制，允许客户端假扮为特定的身份（例如刚刚说的假借SA进行操作的角色），以获取该身份所拥有的权限。
- 访问控制：STS 可以根据访问策略和权限规则控制对临时凭证的颁发和使用，确保安全访问受保护的资源。

STS 通常作为身份提供者和资源提供者之间的中间层，用于协调身份验证和访问控制流程。在云计算云服务提供商中，STS 被广泛应用于安全认证和访问控制方面，为客户端提供安全、可靠的身份验证和访问控制服务。

- 4，CEL

CEL 指 "Common Expression Language"（通用表达式语言），它是一种用于在不同系统之间共享逻辑表达式的标准化语言。它可以用于描述各种计算逻辑，例如筛选、转换、计算等。 CEL 是由 Google 开发的，主要用于在其产品中执行安全的表达式评估，如 Google Cloud IAM（身份和访问管理）中的条件语句和策略评估。 

在 Google Cloud 中，CEL 通常用于编写策略和条件表达式，以便对资源进行访问控制、审计和转换。它允许用户定义灵活的逻辑条件，以便根据特定的规则和要求对资源进行管理和操作。 CEL 支持丰富的语法和功能，包括基本数学运算、逻辑运算、字符串操作、列表和映射处理等。

### recap

其他的更多详细设置参见公式书，以及各个IdP提供商的说明，解决问题的方法总是很多，但是如果只知道详细过程不知道总体原理，每次都不理解每次都是一次性的操作是没法很好的解决问题的。学会阅读公式书也是一个重要的技能。所以很多地方不再赘述，在问题中找到解决问题的办法才是终极答案，没有任何人可以告诉你所有的答案。

我对于认证方面的东西理解还是很碎片，但是通过不断输入和收集，将碎片的理解拼好的感觉很不错。当然系统的学习也很重要。



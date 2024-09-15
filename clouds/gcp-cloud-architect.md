## Management

### Resource Manager

- **GCP构架层**：根为组织，下面是folder层，下面是Project层，pj里面是resources
- PJ的ID和Number是不可变的，全球唯一，Name可以随时变更
- **组织Policy**是和AWS的SCP类似的东西，下面的层级会继承它的Policy
- 创建**组织**不能用个人账户，需要*Google Workspace*或者*Cloud Identity*
- 坚守**最小权限原则**，但是也要尽量削减*管理Overhead*（在管理活动中额外消耗的资源或成本，这些成本并不直接产生效益，但为了确保系统或组织的正常运作而必须承担）
- 环境分离：test，staging，prod，最好是组织层级分离
- 使用*会计组织*管理请求书，其他的组织不拥有阅览权

## Compute

## Storage

## Network

## Database

## Data Analytics

## AI & ML

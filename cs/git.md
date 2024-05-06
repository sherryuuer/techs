## Git & Github 是一种对时间系统的理解

### Git & Github

**VCS（Version Control System，版本控制系统）workflow**

是指在软件开发中使用版本控制系统进行协作和管理代码的一种工作流程。不同的团队和项目可能会采用不同的 VCS workflow，其中最常见的几种包括集中式工作流、分支工作流和分叉工作流。

以下是一些常见的 VCS workflow：

1. **集中式工作流（Centralized Workflow）**：
   - 在集中式工作流中，所有的开发者都从中央仓库（通常是主仓库）获取代码，并将其更改推送回中央仓库。
   - 这种工作流适用于小型团队或简单项目，因为它简单直接，易于理解。

2. **分支工作流（Feature Branch Workflow）**：
   - 在分支工作流中，每个功能或任务都在自己的分支上进行开发，开发完成后再将其合并回主分支。
   - 这种工作流允许开发人员独立开发和测试功能，减少冲突，并且可以方便地跟踪每个功能的进度。

3. **分叉工作流（Forking Workflow）**：
   - 在分叉工作流中，每个开发者都从中央仓库分叉（fork）出自己的仓库，并在自己的仓库中进行开发。然后通过 Pull Request 将更改合并到中央仓库。
   - 这种工作流对于开源项目非常有用，因为它可以为每个贡献者提供独立的工作环境，保持代码库的干净和稳定。

4. **Gitflow 工作流**：
   - Gitflow 是一种基于分支的工作流，它定义了一个严格的分支模型，包括主分支（`master`）、开发分支（`develop`）、功能分支（`feature`）、发布分支（`release`）和修补分支（`hotfix`）等。
   - Gitflow 工作流适用于中大型项目，有助于管理复杂的发布周期和多个并行开发任务。

**ALL GIT REPOSITORIES ARE BORN EQUAL!**

这是一句可爱又强大的话。无法想象没有Git前的代码管理世界。

历史上的源代码管理是集中式的，而现在的Git是分布式的，就像区块链，它让代码实现了去中心化，每个人的代码库都是中心代码的一个完全的copy，不依存也不是影子。这真的是太酷了。

**Branches**

看过洛基剧集了吗，时间分支就是Branches，Git让世界分为更多的支线，每一条时间线都可以有不同的更改，分出不同的故事！

如果有时间跳跃的能力，就可以在不同的时间线之间跳跃，看不同的故事，Git就有这样的能力，我们看到很多软件都有阿尔法，贝塔版本，就像是实验版本和正式版本，这让我们可以用不同版本的代码！

分支的本质就是 A set of changes from a specific point in time.

**Upstream & Downstream**

上游和下游的区别是什么，上游可以简单理解为代码源，下游则是克隆的代码。当你想要将本地Git仓库和fork的Github仓库联动的时候，我们会用这种代码。

比如下面命令组：

```bash
git remote add upstream <forkRepoURL>
git fetch upstream
git merge upstream/main
git push origin <yourbranch>
```

**Git的优势**

和传统的VCS相比，Git的优势在于，分支很便宜，你可以随便添加和删除（当然有些项目的场景需要你保留，不要一刀切）。在很多传统的版本控制系统中，要branch的时间复杂度可能是线性的，这是因为他们很多是以file为基础的。

但是在Git中的时间复杂度是常数时间，因为是基于整个仓库的。

同样的道理，在Git中的commits是基于整个项目的，而传统的系统则是基于文件的。

Git中也没有版本控制号码，取而代之，每一个操作都会被编码一个hash号码。随机，通过这个号码我们可以重新回到历史的每一个commit时间点。

### Git Basics

`git init`

初始化命令，在该文件夹中创建一个`.git`文件夹，这个文件夹就会帮你处理各种版本管理问题。

其中 HEAD 文件记录了你当前所在的分支和 commits ID，config 文件则记录了你的仓库状态。如果你在一个分支，那么HEAD指示你就在这个分支上。

`git log` or `git log --oneline` or `git log --oneline --graph`

记录历史状态的命令。它会访问你的`.git/refs/heads/master`中的历史记录，这种历史记录必须是commit带来的。

加入option可以让log更简略，或者用graph形式显示前后分支和历史的关系。

`git log --decorate --graph --oneline --all`不仅表示了所有详细的信息，还增加了关于分支`--decorate`的信息。

`git status`

这是一个很有用的命令，会帮助告诉你当前git的状态。如果你创建了一个文件，他会告诉你有没有track的文件，如果你add了文件，它会告诉你仓库有了变化。总之它会让你看清现状，这对人和对系统都是一个重要的举动。

`git add file`

add是经常使用的一个命令，它代表文件的更改已经脱离了完全的local，而是进入了一个staging area。

`git commit -m 'message'`

commit意味着终于对本地仓库进行了修改和确认。

`git commit -a -m 'add and commit at the same time'`

如代码中的写的，加入a命令就是合并了add和commit两个功能，懒人必备。

`git diff`

这条指令会告诉你你做了哪些修改。

总的来说整个Git的基础包括了，初始化，回望历史，进行行动，确认行动，不断复盘等过程，完全就是一个人生缩影。

`git clone`和`git reset`

拷贝仓库，和时间回溯。人生没有后悔药，但是Git有。

reset就可以用于灾难恢复。如果你不小心用`rm -rf`删除了所有的仓库文件，但是你的`.git`文件夹还是会存在。就像ls会忽视带点点的文件一样，bash的删除命令也会忽视.git文件夹。

reset 是帮我们进行灾难恢复的重要工具，一般来说我们经常用的 hard 模型，也就是硬恢复，也就是完全的恢复，包括当前的工作目录。

`--soft` 模式只移动 HEAD，`--mixed` 模式移动 HEAD 并取消暂存区的更改，而 `--hard` 模式会移动 HEAD 并重置暂存区和工作目录。

`grep -A2 'remote "origin"' .git/config`

这行指令可以帮助在本地 config 文件中查找关于远程仓库的配置和信息。

`git branch <NewBranchName>`

创建时间分支。

`git checkout <BranchName>`

进入某分支。checkout在英语中是登记，切换的意思。

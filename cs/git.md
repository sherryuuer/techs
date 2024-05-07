## Git & Github 复习手册

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
   - Gitflow 是一种基于分支的工作流，它定义了一个严格的分支模型，包括主分支（`main`）、开发分支（`develop`）、功能分支（`feature`）、发布分支（`release`）和修补分支（`hotfix`）等。
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

另外还有一个`git add -i`是一个交互式的提交修改的模式。使用其中的 patch 等命令可以让 git 逐个询问你的修改是否要被添加，甚至可以分割你的修改内容。个人觉得很复杂，如果不是超级精细的项目管理，应该我暂时不会用到。

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

### Git Branching

`git branch <NewBranchName>`

创建时间分支。

`git checkout <BranchName>`

进入某分支。checkout在英语中是登记，切换的意思。

`git checkout <commitID>`

不仅如此，checkout还可以将HEAD指向任何一个过去的commit点。

`git branch <new-branch-name> <commit-id>`

使用这样的指令，可以在新的分支对过去的commit点进行修改操作和push。

```bash
git checkout e36355ed00ac3af009d7113a9dd281c269a79afd
git branch -f newfeature
```

这两行指令是 Git 中的命令，用于在版本控制中执行特定的操作。

- `git checkout e36355ed00ac3af009d7113a9dd281c269a79afd`：
   - 这条命令的作用是将当前的 HEAD（即当前所在的分支）移动到指定的提交（commit）。在这里，`e36355ed00ac3af009d7113a9dd281c269a79afd` 是一个提交的 SHA-1 标识符，它代表了一个具体的提交。
   - 执行这个命令后，你的工作目录和暂存区将会与这个提交的状态一致，但是你处于一个分离头状态（detached HEAD），不再位于任何一个分支上。
- `git branch -f newfeature`：
   - 这条命令的作用是将一个现有的分支（在这里是 `newfeature`）重新设置到当前 HEAD 所指向的提交。
   - `-f` 选项表示 "force"，表示要强制重置分支的指向，即使这可能会导致丢失一些提交。在这里，它将 `newfeature` 分支强制设置到当前 HEAD 所指向的提交上。
   - 执行这个命令后，`newfeature` 分支将会指向与当前 HEAD 相同的提交，这样你就可以继续在 `newfeature` 分支上进行开发工作，而不必担心丢失之前的提交。

`git tag remember_to_tell_bob_to_rewrite_this newfeature`

这条命令的作用是创建一个新的标签，并将其附加到指定的提交上。在这里，新标签的名称是 remember_to_tell_bob_to_rewrite_this，它指向的提交是 newfeature 分支当前所指向的提交。标签通常用于标记重要的提交，比如版本发布或里程碑。在这里，这个标签的名称似乎是一个提醒，表示要告诉 Bob 重写这个提交，可能是因为提交有问题或者需要进一步修改。

使用标签可以方便地引用特定的提交，也可以在团队协作中作为提醒或者注释使用。

### Merging

Merging 其实就是 Branching 的相反操作，就像是河流会分叉，然后又会融合。

Conflicts 冲突，可能是 Merge 操作中最难搞的部分。它是指在不同的分支中，比如 main 和一个 feature 分支中对同一个文件做了修改，当要进行 merge 的时候，系统会提示冲突了。当两个人对同一个文件的逻辑都进行了修改，如果随便合并，就可能逻辑错误了，Git可不是随随便便的人。

这个时候冲突的文件中会用很多小箭头记录你的冲突点，你修改文件后手动commit就可以了，甚至你不修改就那么放着然后commit，Git也不会在意。最后用`git log --all --oneline --graph --decorate`查看记录，就会看到你的修改历史了。（注意log的表示是从新到旧的。）

`git merge -X ours -m merged abranch`

这行代码的含义是执行 Git 合并操作，使用了 `-X ours` 选项来确保在合并冲突时优先保留当前分支的更改，同时指定了合并操作的提交信息为 “merged”，并且将名为 “abranch” 的分支合并到当前分支。使用这种方法也可以解决冲突。

### Git Stash

`git stash` 命令用于临时保存当前工作目录的修改，并将工作目录恢复到干净的状态，以便你可以在稍后的时间重新应用这些修改。它的主要作用是帮助你在切换分支或者处理其他任务时，暂时存储当前的工作进度。

通常情况下，当你在工作目录中有一些修改，但是还没有准备好提交它们时，需要切换到另一个分支，或者解决一些紧急问题时，使用 `git stash` 命令非常方便。

下面是 `git stash` 命令的一些常见用法和选项：

- `git stash`：保存当前工作目录的修改到一个临时的存储区域，并将工作目录恢复到干净的状态。
- `git stash save "message"`：保存当前工作目录的修改，并附带一条消息，用于描述这次保存的内容。
- `git stash list`：列出当前保存的所有 stash 记录。
- `git stash apply`：将最近保存的 stash 应用到当前工作目录，但不会从 stash 列表中移除该记录。
- `git stash pop`：将最近保存的 stash 应用到当前工作目录，并从 stash 列表中移除该记录。
- `git stash drop stash@{n}`：移除指定的 stash 记录。
- `git stash clear`：移除所有保存的 stash 记录。

使用 `git stash` 命令时，Git 会将当前的修改保存到一个堆栈中，并给每个保存的修改记录一个索引（类似于 `stash@{0}`、`stash@{1}`）。当你想要重新应用这些修改时，可以使用 `git stash apply stash@{1}` 或者 `git stash pop` 命令来将这些修改重新应用到工作目录中。

看到pop我们就知道了这些更改被存放在了一个stack中。这个stack是git的另一个分支。

想象一个流程，当你正在做修改，然后接到了别的任务，于是首先 git diff 确认你修改的地方，然后 git stash 存储修改顺便清理了环境，通过 git status 就可以确认这一点，当你完成了别的工作后，回来进行 git stash pop 拿回了刚刚存储的修改。

### Git Reflog

`git reflog` 命令用于查看引用日志（reference log）。它记录了 HEAD 和分支指针的移动历史，以及每次操作的提交哈希值。主要用于恢复丢失的提交或分支，或者查看在本地仓库中执行的所有引用更新操作，包括分支移动、分支删除等。

所以和`git log`的用途不同，`git reflog`主要用于恢复丢失的提交部分。

确定了需要回到的commit点，使用`git reset --hard 40e99f7`进行恢复即可。

### Cherry-Pick

挑樱桃，怎么会有这么可爱的功能w

Cherry-picking 是 Git 中的一种操作，用于选择性地将一个或多个提交从一个分支应用到另一个分支上。它的名字来源于挑选樱桃的行为，即从一个分支上挑选出你需要的提交，然后应用到另一个分支上，就像挑选樱桃一样选择性地挑选提交。

Cherry-picking 通常用于将某些特定的提交应用到当前分支，而不需要将整个分支合并过来。这对于在不同分支上开发不同功能的情况下，想要将特定的修改应用到其他分支上非常有用。

```bash
git cherry-pick <commit-hash>
```

这会将指定提交的修改应用到当前分支上，并创建一个新的提交来代表这个变更。如果你想要将多个提交应用到当前分支，可以依次指定多个提交的哈希值。

Cherry-picking 的过程会尝试将选定的提交应用到当前分支上，如果遇到冲突，需要手动解决冲突并完成 cherry-pick 操作。

### Git Rebase

Git 中的 rebase 是一种常见的操作，用于将一个分支的提交移动到另一个分支上，以便保持历史记录的整洁和线性，并且减少不必要的合并。rebase 操作会将当前分支的提交“重新定位”到目标分支的最新提交之后。

通常情况下，rebase 操作被用来实现以下几个目的：

rebase 操作的基本用法是：

```bash
git checkout <target_branch>
git pull origin <target_branch>  # 更新目标分支
git checkout <source_branch>
git rebase <target_branch>
```

上述命令将先切换到目标分支，然后更新目标分支以获取最新的提交，然后切换回源分支，并将源分支的提交基于目标分支进行 rebase。

这样一来主分支的HEAD就指向了目标分支的最后一个提交了。

上面这个是在main分支上rebase目标分支。那么如果反过来呢？

如果直接在某分支上执行 `git rebase main`，则会将当前分支的提交基于 master 分支进行重新定位。这意味着 Git 会首先找到当前分支和 master 分支的分叉点，然后将当前分支的提交逐个应用到 main 分支的最新提交之后。

这种操作的结果同样是，当前分支的提交历史会在 main 分支的最新提交之后重新生成，从而保持了整洁的历史记录和线性的提交序列。

需要注意的是，执行 rebase 操作可能会产生冲突，特别是当当前分支的提交与 main 分支的提交产生了冲突时。在这种情况下，Git 会暂停 rebase 进程，并提示你解决冲突。你需要手动解决冲突，然后使用 `git rebase --continue` 命令继续 rebase 进程。

总的来说，两个方法的结果是一样的。

**Fast Forward**

这个时候就很容易理解这种HEAD指针简单易懂的行为了。在我们进行 merge 操作的时候，有时候内部进行的就是默认的fast forward。

它表示如果要合并的分支的提交历史中包含了目标源分支的所有提交，且要合并分支的提交是目标源分支的一个祖先提交，那么 Git 将执行 fast-forward 合并，直接将目标源分支指向要合并分支的最新提交。

在执行这些步骤后，目标源分支的指针会直接移动到要合并分支的最新提交，完成 fast-forward 合并。这种合并方式通常会保持提交历史的整洁和线性，不会产生额外的合并提交。

这种方式虽然是 merge 指令在某些情况下内部默认的，但是和rebase的工作方式是很像的。

### Git Bisect

`git bisect` 是 Git 提供的一个用于二分查找（binary search）定位代码引入错误的工具。它可以帮助你快速地确定出现错误的具体提交，从而更容易地找到错误引入的原因。

使用 `git bisect` 的过程通常如下：

首先你要定位查找范围，也就是最早的你觉得是good的提交，比如通过指令`git checkout HEAD~99`回到99个提交之前。以及你发现有问题的提交点为bad。那么现在的目标就是找到这中间的，第一个引发bug的坏提交。

运行 `git bisect start` 命令，开始 bisect 进程。在过程中，使用 `git bisect bad` 或者 `git bisect good` 命令告诉 Git 当前的提交是bad还是good。

Git 会根据你提供的信息选择一个中间的提交作为下一个要测试的提交。在每次 bisect 过程中，你需要测试当前提交，看是否存在错误。重复该步骤直到 Git 找到引入错误的提交为止。

当 Git 找到引入错误的提交后，它会停止 bisect 进程，并告诉你引入错误的具体提交。

### Clone Repo

Origin 是一个业界默认的名字，代表remote，也就是远程仓库，也可以是从本地克隆的，所以说叫源头仓库可能更好。

fetch 和 push 是针对remote仓库的两种常见操作。

`git pull` = `git fetch origin main` + `git merge origin/main`

也就是说fetch 的结果，是将远程的修改，取回到本地的一个叫`origin/main`的分支上。虽然常用 pull，但是通过 fetch 可以更好的理解背后发生了什么。

这也意味着我们不一定非要把origin设置为唯一的remote仓库，我么还可以设置更多的合作伙伴的remote仓库，进行拉取操作。比如：

`git remote add alice alice_repository` & `git fetch alice master` & `git merge alice/master`。

毕竟每个repo生来平等。


### Git Push

`git push` 将操作推送到远程分支，如果是从本地branch推送，并且远程没有该分支，则需要`git push origin branch_name`来自动为远程仓库创建一个新的分支进行推送。

如果你要推送的目的地和你的本地分支内容有冲突，则需要先进行 fetch，然后 merge（或者直接pull），合并了远程的新内容之后，再进行push。

如果你远程的分支，本地没有怎么办，fetch可以办到。

fetch 不仅可以取回远程的内容，还可以取回远程的分支。使用指令：`git fetch origin`就可以将远程的所有分支拿回本地了。

但是这个时候你用`git branch`命令是无法找到分支的，使用`git branch -a`可以发现所有的分支在remotes/origin中。

```bash
$ git branch -a
* master
  remotes/origin/HEAD -> origin/master
  remotes/origin/abranch
  remotes/origin/master
```

这个时候 checkout，git 就会帮你自动 track 远程的分支了。

```bash
$ git checkout abranch
Branch abranch set up to track remote branch abranch from origin.
Switched to a new branch 'abranch'
$ git branch
* abranch
  master
```

反过来，如果你本地的分支远程没有怎么办，这个情况很常用。因为我们经常要在本地创建新的分支。使用如下命令就可以为远程创建分支和自动track了。

`git push --set-upstream origin abranch` or `git push -u origin abranch`

注意`git push origin abranch`可以推送，但是不会自动track远程分支。所以使用上面的两行比较方便。

### Git Submodules

Git Submodules 是 Git 版本控制系统中的一种功能，允许你将一个 Git 仓库作为另一个 Git 仓库的子目录引入。这对于管理依赖关系、子模块的版本和跟踪外部代码库的变化非常有用。

使用 Git Submodules 可以将一个或多个外部仓库嵌入到你的项目中，并保持它们独立的版本历史。这意味着你可以在主项目中使用外部仓库的特定版本，并且可以随时更新或切换到新的外部仓库版本。

以下是一些常用的 Git Submodules 相关指令：

`git submodule add <repository-url> <path>`：将外部仓库添加为你的项目的子模块。

`git submodule init`：克隆主项目时，如果主项目包含子模块，需要初始化子模块。

`git submodule update`：获取子模块的最新代码。

`git clone --recurse-submodules <repository-url>`：克隆主项目及其子模块。

在子模块中切换到特定分支或版本：

```
cd <path/to/submodule>
git checkout <branch/tag/commit>
```

从项目中移除子模块：
```
git submodule deinit <path>
git rm <path>
```

如果你对子模块进行了更改，需要提交这些更改并更新主项目：
```
git submodule update --remote
git add <path/to/submodule>
git commit -m "Update submodule"
```
这部分不是很常用，有记忆点就好。

### Pull Requests

PR不是Git的核心模块，很多不同的应用PR的细节不同，我们这里专注于Github即可。

标准的 GitHub 模型是：

- fork别人的repository
- 将这个fork的repo克隆到自己的本地
- 本地的repo上创建一个分支
- 在此分支上进行更改并推送到自己的远程fork的仓库
- 向原始存储库发出拉取请求以合并此分支的更改

在发出请求的时候，审核者经常会要求将分支重新定位到主分支。这样更有利于代码审查，和解决冲突，以及历史一致性。

首先更新主分支：

```bash
git checkout master
git pull origin master
```

然后切换到分支，并rebase到主分支：

```bash
git checkout your-feature-branch
git rebase master
```

如果有冲突那么解决冲突继续rebase：

```bash
# 解决冲突后，将文件标记为已解决
git add <conflicted-file>

# 继续 rebase 进行操作
git rebase --continue
```

然后推送分支后，就可以进行PR了。

```bash
git push origin your-feature-branch --force
```

在开源项目中这是标配！

### Git Log flags

`--oneline`

可以只显示commit id和信息，而隐藏作者和邮件。

`--oneline --graph`

用图的形式显示历史，从新到旧。

`--graph --oneline --all`

不止显示HEAD所在的分支，而是所有分支的历史。

`--graph --oneline --all --decorate`

加上decorate就可以显示分支的名称了。

`-–simplify-by-decoration`

用于简化输出的提交历史。其作用是根据标签（tags）或分支（branches）来简化历史记录的显示。

`--graph --oneline --all --decorate --simplify-by-decoration --pretty='%ar %s %h'`

--pretty 可以优化日期的输出格式。成为如下形式：

```bash
* 6 years ago release: 1.0.153 ad1a789
* 6 years ago release: 1.0.152 3f6990b
* 6 years ago release: 1.0.151 381afdb
...
```

### Squashing commits 

是指将多个连续的 Git 提交合并成一个单独的提交。这个过程会将一系列相关的提交整合成一个更大的提交，以便更清晰地记录项目历史或减少提交历史中的噪音。

使用指令`git rebase -i`

具体来说需要一个进行合并的最老的commit，和一个最新的commit。一般来说默认是现在的HEAD，然后执行上述命令后，可以进行一个交互式操作，对上一次推送以来的commit进行合并操作。保存交互式界面后，git就会按照记录进行操作了。

重点就是对每一行commit进行操作指令的修改，修改后的文件就像是对 Git 的一个指令集。

### Git Hooks

Git Hooks 是 Git 版本控制系统中的一种机制，允许你在特定的 Git 操作发生时执行自定义脚本或命令。这些操作可以是提交、合并、推送等等，Git Hooks 可以帮助你在这些操作发生时自动化执行额外的任务或操作。

Git Hooks 存储在 Git 仓库的 `.git/hooks/` 目录中，每个 Git 钩子都是一个可执行文件，通常是 shell 脚本或任何可执行的程序。

以下是一些常见的 Git Hooks：

1. **pre-commit**：在执行提交操作之前触发，允许你在提交之前运行自定义的代码，例如代码格式化、代码风格检查或单元测试等。

2. **prepare-commit-msg**：在提交消息被编辑器调用之前触发，允许你修改或自动生成提交消息。

3. **post-commit**：在提交操作完成后触发，允许你在提交后执行任何必要的操作，例如发送通知或执行其他脚本。

4. **pre-receive**：在接收远程推送之前触发，允许你在远程推送到仓库之前执行验证、审查或其他操作。

5. **post-receive**：在接收远程推送之后触发，允许你在推送完成后执行任何必要的操作，例如触发持续集成流程或更新相关文档。

6. **pre-push**：在执行推送操作之前触发，允许你在推送之前运行自定义的代码，例如运行测试套件或代码质量检查。

等等。

通过编写自定义的 Git Hooks 脚本，你可以根据项目的特定需求添加自动化操作，提高开发效率、代码质量和协作流程的规范性。


---
参考书中文版 [gitbook](https://gitbook.liuhui998.com) 比较好。


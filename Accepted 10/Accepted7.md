# Challenge Accepted 10 2020.4.16 周五


[19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

[83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

[226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

[78. 子集](https://leetcode-cn.com/problems/subsets/)

[31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)

[93. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses/)

[165. 比较版本号](https://leetcode-cn.com/problems/compare-version-numbers/)

[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

[62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

[101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)



------





[39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gplqpjg5wdj31ax0jnn10.jpg)

```go
func combinationSum(candidates []int, target int) (res [][]int) {
	path := []int{}
	sort.Ints(candidates)
	var dfs func(int, int)

	dfs = func(target, index int) {
		if target <= 0 {
			if target == 0 {
				res = append(res, append([]int(nil), path...))
			}
			return
		}
		for i := index; i < len(candidates); i++ { // 枚举当前可选的数，从index开始
			if candidates[i] > target { // 剪枝优化
				break
			}
			path = append(path, candidates[i]) // 选这个数,基于此，继续选择，传i，下次就不会选到i左边的数
			dfs(target-candidates[i], i)       // 注意这里迭代的时候 index 依旧不变，因为一个元素可以取多次
			path = path[:len(path)-1]          // 撤销选择，回到选择candidates[i]之前的状态，继续尝试选同层右边的数
		}
	}

	dfs(target, 0)
	return
}

```

[64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

### 方法一：动态规划

```go
func minPathSum(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	for i := 1; i < m; i++ {
		grid[i][0] += grid[i-1][0]
	}
	for j := 1; j < n; j++ {
		grid[0][j] += grid[0][j-1]
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			grid[i][j] += min(grid[i][j-1], grid[i-1][j])
		}
	}
	return grid[m-1][n-1]
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```

复杂度分析

- 时间复杂度：O(mn)，其中 m 和 n 分别是网格的行数和列数。需要对整个网格遍历一次，计算 dp 的每个元素的值。

- 空间复杂度：O(1)



[239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

### 方法一 暴力解法 O(nk)

```go
func maxSlidingWindow(nums []int, k int) []int {
	res, n := make([]int, 0, k), len(nums)
	if n == 0 {
		return make([]int, 0)
	}
	for i := 0; i <= n-k; i++ {
		max := nums[i]
		for j := 1; j < k; j++ {
			if max < nums[i+j] {
				max = nums[i+j]
			}
		}
		res = append(res, max)
	}
	return res
}
```
Time Limit Exceeded
50/61 cases passed (N/A)

### 方法二 双端队列 Deque
最优的解法是用双端队列，队列的一边永远都存的是窗口的最大值，队列的另外一个边存的是比最大值小的值。队列中最大值左边的所有值都出队。在保证了双端队列的一边即是最大值以后，时间复杂度是 O(n)，空间复杂度是 O(K)

```go
func maxSlidingWindow(nums []int, k int) []int {
	if len(nums) == 0 || len(nums) < k {
		return make([]int, 0)
	}
	window := make([]int, 0, k) // store the index of nums
	result := make([]int, 0, len(nums)-k+1)
	for i, v := range nums {
		if i >= k && window[0] <= i-k { // if the left-most index is out of window, remove it
			window = window[1:len(window)]
		}
		for len(window) > 0 && nums[window[len(window)-1]] < v {
			window = window[0 : len(window)-1]
		}
		window = append(window, i)
		if i >= k-1 {
			result = append(result, nums[window[0]]) // the left-most is the index of max value in nums
		}
	}
	return result
}
```

![](http://ww1.sinaimg.cn/large/007daNw2ly1gpmcfyuvh1j319g0mun0a.jpg)

```go
func maxSlidingWindow(nums []int, k int) []int {
	q, res := []int{}, []int{}
	for i := 0; i < len(nums); i++ {
		if len(q) > 0 && i-k+1 > q[0] {
			q = q[1:] //窗口满了，删除队头
		}
		for len(q) > 0 && nums[q[len(q)-1]] <= nums[i] {
			q = q[:len(q)-1] //队尾小于当前元素，删除队尾
		}
		q = append(q, i)
		if i >= k-1 { //窗口大小大于等于 k
			res = append(res, nums[q[0]])
		}
	}
	return res
}
```

[78. 子集](https://leetcode-cn.com/problems/subsets/)


### 方法一：位运算

![截屏2021-04-17 11.45.32.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpmlgt89mej31120o2wgb.jpg)

```go
func subsets(nums []int) [][]int {
	res, n := [][]int{}, len(nums)
    //1<<3 二进制：1000 十进制：1*2^n=8
	for i := 0; i < 1<<n; i++ { // i 从 000 到 111 
		tmp := []int{}
		for j := 0; j < n; j++ {
			if i>>j&1 == 1 { // i 的第 j 位是否为1
				tmp = append(tmp, nums[j])
			}
		}
		res = append(res, tmp)
	}
	return res
}
```

### 方法一：迭代法实现子集枚举
思路与算法

记原序列中元素的总数为 n。原序列中的每个数字 ai 的状态可能有两种，即「在子集中」和「不在子集中」。我们用 1 表示「在子集中」，0 表示不在子集中，那么每一个子集可以对应一个长度为 n 的 0/1 序列，第 i 位表示 ai 是否在子集中。
例如，n=3 ，a={5,2,9} 时：

![截屏2021-04-17 14.57.34.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpmr0nopx0j316u0judho.jpg)

可以发现 0/1 序列对应的二进制数正好从 0 到 2^n - 1。我们可以枚举 mask∈[0, 2^n - 1]，mask 的二进制表示是一个 0/1 序列，我们可以按照这个 0/1 序列在原集合当中取数。当我们枚举完所有 2^n 个 mask，我们也就能构造出所有的子集。

```go
func subsets(nums []int) [][]int {
	res, n := [][]int{}, len(nums)
	for mask := 0; mask < 1<<n; mask++ {
		set := []int{}
		for i, v := range nums {
			if mask>>i&1 == 1 {
				set = append(set, v)
			}
		}
		res = append(res, append([]int(nil), set...))
	}
	return res
}
```

复杂度分析

- 时间复杂度：O(n×2^n)。一共 2^n 个状态，每种状态需要 O(n) 的时间来构造子集。

- 空间复杂度：O(n)。即构造子集使用的临时数组 t 的空间代价。



### 方法二：递归法实现子集枚举

#### 思路 1
- 单看每个元素，都有两种选择：选入子集，或不选入子集。
- 比如[1,2,3]，先看1，选1或不选1，都会再看2，选2或不选2，以此类推。
- 考察当前枚举的数，基于选它而继续，是一个递归分支；基于不选它而继续，又是一个分支。

![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpms453jltj31fw0jjq6r.jpg)

- 用索引index代表当前递归考察的数字nums[index]。
- 当index越界时，所有数字考察完，得到一个解，位于递归树的底部，把它加入解集，结束当前递归分支。

#### 为什么要回溯？
- 因为不是找到一个子集就完事。
- 找到一个子集，结束递归，要撤销当前的选择，回到选择前的状态，做另一个选择——不选当前的数，基于不选，往下递归，继续生成子集。
- 回退到上一步，才能在包含解的空间树中把路走全，回溯出所有的解。

![2.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpms4bgnnkj31a60lc0w8.jpg)


```go
func subsets(nums []int) [][]int {
	res, set := [][]int{}, []int{}
	var dfs func(int)

	dfs = func(i int) {
		if i == len(nums) { // 指针越界
			res = append(res, append([]int(nil), set...)) // 加入解集
			return                                        // 结束当前的递归
		}
		set = append(set, nums[i]) //选择这个数
		dfs(i + 1)                 // 基于该选择，继续往下递归，考察下一个数
		set = set[:len(set)-1]     // 上面的递归结束，撤销该选择
		dfs(i + 1)                 // 不选这个数，继续往下递归，考察下一个数
	}

	dfs(0)
	return res
}
```

#### 思路2
刚才的思路是：逐个考察数字，每个数都选或不选。等到递归结束时，把集合加入解集。
换一种思路：在执行子递归之前，加入解集，即，在递归压栈前 “做事情”。

![3.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpms4h4i9rj31070h1mz8.jpg)

- 用 for 枚举出当前可选的数，比如选第一个数时：1、2、3 可选。
1. 如果第一个数选 1，选第二个数，2、3 可选；
2. 如果第一个数选 2，选第二个数，只有 3 可选（不能选1，产生重复组合）
3. 如果第一个数选 3，没有第二个数可选
- 每次传入子递归的 index 是：当前你选的数的索引+1当前你选的数的索引+1。
- 每次递归枚举的选项变少，一直递归到没有可选的数字，进入不了for循环，落入不了递归，整个DFS结束。
- 可见我们没有显式地设置递归的出口，而是通过控制循环的起点，使得最后递归自然结束。

```go
func subsets(nums []int) [][]int {
	res, set := [][]int{}, []int{}
	var dfs func(int)

	dfs = func(i int) {
		res = append(res, append([]int(nil), set...)) // 调用子递归前，加入解集
		for j := i; j < len(nums); j++ {              // 枚举出所有可选的数
			set = append(set, nums[j]) // 选这个数
			dfs(j + 1)                 // 基于选这个数，继续递归，传入的j+1，不是i+1
			set = set[:len(set)-1]     // 撤销选这个数
		}
	}

	dfs(0)
	return res
}
```


[31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)

### 思路

1. 我们需要将一个左边的「较小数」与一个右边的「较大数」交换，以能够让当前排列变大，从而得到下一个排列。

2. 同时我们要让这个「较小数」尽量靠右，而「较大数」尽可能小。当交换完成后，「较大数」右边的数需要按照升序重新排列。这样可以在保证新排列大于原来排列的情况下，使变大的幅度尽可能小。


- 从低位挑一个大一点的数，换掉前面的小一点的一个数，实现变大。
- 变大的幅度要尽量小。


像 [3,2,1] 递减的，没有下一个排列，因为大的已经尽量往前排了，没法更大。

像 [1,5,2,4,3,2] 这种，我们希望它稍微变大。

从低位挑一个大一点的数，换掉前面的小一点的一个数。

于是，从右往左，寻找第一个比右邻居小的数。（把它换到后面去）

找到 1 5 (2) 4 3 2 中间这个 2，让它和它身后的一个数交换，轻微变大。

还是从右往左，寻找第一个比这个 2 微大的数。15 (2) 4 (3) 2，交换，变成 15 (3) 4 (2) 2。

这并未结束，变大的幅度可以再小一点，仟位变大了，后三位可以再小一点。

后三位是递减的，翻转，变成[1,5,3,2,2,4]，即为所求。



```go
func nextPermutation(nums []int) {
	i := len(nums) - 2                   // 向左遍历，i从倒数第二开始是为了nums[i+1]要存在
	for i >= 0 && nums[i] >= nums[i+1] { // 寻找第一个小于右邻居的数
		i--
	}
	if i >= 0 { // 这个数在数组中存在，从它身后挑一个数，和它换
		j := len(nums) - 1                 // 从最后一项，向左遍历
		for j >= 0 && nums[j] <= nums[i] { // 寻找第一个大于 nums[i] 的数
			j--
		}
		nums[i], nums[j] = nums[j], nums[i] // 两数交换，实现变大
	}
	// 如果 i = -1，说明是递减排列，如 3 2 1，没有下一排列，直接翻转为最小排列：1 2 3
	l, r := i+1, len(nums)-1
	for l < r { // i 右边的数进行翻转，使得变大的幅度小一些
		nums[l], nums[r] = nums[r], nums[l]
		l++
		r--
	}
}
```

[165. 比较版本号](https://leetcode-cn.com/problems/compare-version-numbers/)

```go
func compareVersion(s1 string, s2 string) int {
	i, j := 0, 0
	for i < len(s1) || j < len(s2) {
		a, b := "", ""
		for i < len(s1) && s1[i] != '.' {
			a += string(s1[i])
			i++
		}
		for j < len(s2) && s2[j] != '.' {
			b += string(s2[j])
			j++
		}
		x, _ := strconv.Atoi(a) //string 转 int
		y, _ := strconv.Atoi(b)
		if x > y {
			return 1
		} else if x < y {
			return -1
		}
		i++
		j++
	}
	return 0
}
```

```go
strconv.Atoi()函数用于将字符串类型的整数转换为int类型，函数签名如下。

func Atoi(s string) (i int, err error)


strconv.Itoa()函数用于将int类型数据转换为对应的字符串表示，具体的函数签名如下。

func Itoa(i int) string
```



[226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

### 方法一：dfs 递归

#### 递归思路1

我们从根节点开始，递归地对树进行遍历，并从叶子结点先开始翻转。如果当前遍历到的节点 root 的左右两棵子树都已经翻转，那么我们只需要交换两棵子树的位置，即可完成以 root 为根节点的整棵子树的翻转。

*思路*
一个二叉树，怎么才算翻转了？

它的左右子树要交换，并且左右子树内部的所有子树，都要进行左右子树的交换。

![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpmz43wk4yj31er0fw0w1.jpg)


每个子树的根节点都说：先交换我的左右子树吧。那么递归就会先压栈压到底。然后才做交换。
即，位于底部的、左右孩子都是 null 的子树，先被翻转。
随着递归向上返回，子树一个个被翻转……整棵树翻转好了。
问题是在递归出栈时解决的。

```go
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	invertTree(root.Left)
	invertTree(root.Right)
	root.Left, root.Right = root.Right, root.Left
	return root
}
```

#### 递归思路 2

![2.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpmz4kjl1jj31fu0gh77e.jpg)

思路变了：先 “做事”——先交换左右子树，它们内部的子树还没翻转——丢给递归去做。
把交换的操作，放在递归子树之前。
问题是在递归压栈前被解决的。

```go
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	root.Left, root.Right = root.Right, root.Left
	invertTree(root.Left)
	invertTree(root.Right)
	return root
}
```

复杂度分析

- 时间复杂度：O(N)，其中 N 为二叉树节点的数目。我们会遍历二叉树中的每一个节点，对每个节点而言，我们在常数时间内交换其两棵子树。

- 空间复杂度：O(N)。使用的空间由递归栈的深度决定，它等于当前节点在二叉树中的高度。在平均情况下，二叉树的高度与节点个数为对数关系，即 O(logN)。而在最坏情况下，树形成链状，空间复杂度为 O(N)。



#### 总结
两种分别是后序遍历和前序遍历。都是基于DFS，都是先遍历根节点、再遍历左子树、再右子树。
唯一的区别是：
前序遍历：将「处理当前节点」放到「递归左子树」之前。
后序遍历：将「处理当前节点」放到「递归右子树」之后。

这个「处理当前节点」，就是交换左右子树 ，就是解决问题的代码：

```go
root.Left, root.Right = root.Right, root.Left
```

递归只是帮你遍历这棵树，核心还是解决问题的代码，递归把它应用到每个子树上，解决每个子问题，最后解决整个问题。

### 方法二：BFS 

用层序遍历的方式去遍历二叉树。

根节点先入列，然后出列，出列就 “做事”，交换它的左右子节点（左右子树）。
并让左右子节点入列，往后，这些子节点出列，也被翻转。
直到队列为空，就遍历完所有的节点，翻转了所有子树。

解决问题的代码放在节点出列时。


```go
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	q := []*TreeNode{root}
	for len(q) > 0 {
		cur := q[0]
		q = q[1:len(q)]
		cur.Left, cur.Right = cur.Right, cur.Left
		if cur.Left != nil {
			q = append(q, cur.Left)
		}
		if cur.Right != nil {
			q = append(q, cur.Right)
		}
	}
	return root
}
```


[165. 比较版本号](https://leetcode-cn.com/problems/compare-version-numbers/)

```go
func compareVersion(s1 string, s2 string) int {
	i, j := 0, 0
	for i < len(s1) || j < len(s2) {
		a, b := "", ""
		for i < len(s1) && s1[i] != '.' {
			a += string(s1[i])
			i++
		}
		for j < len(s2) && s2[j] != '.' {
			b += string(s2[j])
			j++
		}
		x, _ := strconv.Atoi(a)
		y, _ := strconv.Atoi(b)
		if x > y {
			return 1
		} else if x < y {
			return -1
		}
		i++
		j++
	}
	return 0
}

```



[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

### 思路
1. 有序数组分成了左右2个小的有序数组，而实际上要找的是右边有序数组的最小值
2. 如果中间值大于右边的最大值，说明中间值还在左边的小数组里，需要left向右移动
3. 否则就是中间值小于等于当前右边最大值，mid 已经在右边的小数组里了，但是至少说明了当前右边的right值不是最小值了或者不是唯一的最小值，需要慢慢向左移动一位。


```go
func findMin(nums []int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] > nums[right] {
			left = mid + 1
		} else {
			right--
		}
	}
	return nums[left]
}
```

```go
func findMin(nums []int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := left + (right-left)>>1
		if nums[mid] < nums[right] {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return nums[left]
}
```



[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

### 方法一：二分查找

### 解题思路 
- 给出一个有序数组 nums 和一个数 target，要求在数组中找到第一个和这个元素相等的元素下标，最后一个和这个元素相等的元素下标。

- 这一题是经典的二分搜索变种题。二分搜索有 4 大基础变种题：

	1. 查找第一个值等于给定值的元素
	2. 查找最后一个值等于给定值的元素
	3. 查找第一个大于等于给定值的元素
	4. 查找最后一个小于等于给定值的元素
这一题的解题思路可以分别利用变种 1 和变种 2 的解法就可以做出此题。或者用一次变种 1 的方法，然后循环往后找到最后一个与给定值相等的元素。不过后者这种方法可能会使时间复杂度下降到 O(n)，因为有可能数组中 n 个元素都和给定元素相同。(4 大基础变种的实现见代码)

```go
func searchRange(nums []int, target int) []int {
	return []int{searchFirstEqualElement(nums, target), searchLastEqualElement(nums, target)}
}

// 二分查找第一个与 target 相等的元素，时间复杂度 O(logn)
func searchFirstEqualElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] < target {
			left = mid + 1
		} else if nums[mid] > target {
			right = mid - 1
		} else {
			if mid == 0 || nums[mid-1] != target { // 找到第一个与 target 相等的元素
				return mid
			}
			right = mid - 1
		}
	}
	return -1
}

// 二分查找最后一个与 target 相等的元素，时间复杂度 O(logn)
func searchLastEqualElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] < target {
			left = mid + 1
		} else if nums[mid] > target {
			right = mid - 1
		} else {
			if mid == len(nums)-1 || nums[mid+1] != target { // 找到最后一个与 target 相等的元素
				return mid
			}
			left = mid + 1
		}
	}
	return -1
}

// 二分查找第一个大于等于 target 的元素，时间复杂度 O(logn)
func searchFirstGreaterElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] >= target {
			if mid == 0 || nums[mid-1] < target { // 找到第一个大于等于 target 的元素
				return mid
			}
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return -1
}

// 二分查找最后一个小于等于 target 的元素，时间复杂度 O(logn)
func searchLastLessElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] <= target {
			if mid == len(nums)-1 || nums[mid+1] > target { // 找到最后一个小于等于 target 的元素
				return mid
			}
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}
```
### 方法二：二分查找

```go
func searchRange(nums []int, target int) []int {
	leftmost := sort.SearchInts(nums, target)
	if leftmost == len(nums) || nums[leftmost] != target {
		return []int{-1, -1}
	}
	rightmost := sort.SearchInts(nums, target+1) - 1
	return []int{leftmost, rightmost}
}
```

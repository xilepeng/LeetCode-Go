
## 任何梦想成功的人最后会发现，要达到这个目的，必须以不平凡的方式来生活，除此之外，别无他法。

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



[62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

[101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

[226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

[72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

[93. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses/)


[剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)








[48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

[31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)


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



[958. 二叉树的完全性检验](https://leetcode-cn.com/problems/check-completeness-of-a-binary-tree/)


[136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

[129. 求根节点到叶节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)


[82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)


[剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)
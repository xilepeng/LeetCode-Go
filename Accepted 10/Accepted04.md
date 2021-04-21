
[剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

[42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

[69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

[704. 二分查找](https://leetcode-cn.com/problems/binary-search/)

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)  补充

[70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

[151. 翻转字符串里的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

[23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

[104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

[2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

[110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)


------



[剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

```go
func getKthFromEnd(head *ListNode, k int) *ListNode {
     slow, fast := head, head 
     for ; k>0; k-- {
         fast = fast.Next
     }
     for fast != nil {
         slow, fast = slow.Next, fast.Next
     }
     return slow
}
```



[42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

### 方法一：双指针

![截屏2021-04-06 18.24.23.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpa760w1vwj31440gy77h.jpg)

```go
func trap(height []int) (res int) {
	left, right := 0, len(height)-1
	leftMax, rightMax := 0, 0
	for left < right {
		leftMax = max(leftMax, height[left])
		rightMax = max(rightMax, height[right])
		if height[left] < height[right] {
			res += leftMax - height[left]
			left++
		} else {
			res += rightMax - height[right]
			right--
		}
	}
	return
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

### 方法二：单调栈
```go
func trap(height []int) (res int) {
	stack := []int{}
	for i, h := range height {
		for len(stack) > 0 && h > height[stack[len(stack)-1]] {
			top := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				break
			}
			left := stack[len(stack)-1]
			curWidth := i - left - 1
			curHeight := min(height[left], h) - height[top]
			res += curWidth * curHeight
		}
		stack = append(stack, i)
	}
	return
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```

单调栈：查找每个数左侧第一个比它小的数
单调队列：滑动窗口中的最值





[69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

### 方法一：二分查找

```go
func mySqrt(x int) (res int) {
	left, right := 0, x
	for left <= right {
		mid := left + (right-left)>>1
		if mid*mid <= x {
			res = mid
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return 
}
```
复杂度分析

- 时间复杂度：O(logx)，即为二分查找需要的次数。
- 空间复杂度：O(1)。

### 方法二：牛顿迭代

```go
func mySqrt(x int) int {
	r := x
	for r*r > x {
		r = (r + x/r) >> 1
	}
	return r
}
```
复杂度分析

- 时间复杂度：O(logx)，此方法是二次收敛的，相较于二分查找更快。
- 空间复杂度：O(1)。






[704. 二分查找](https://leetcode-cn.com/problems/binary-search/)

1. 如果目标值等于中间元素，则找到目标值。
2. 如果目标值较小，继续在左侧搜索。
3. 如果目标值较大，则继续在右侧搜索。

#### 算法：

- 初始化指针 left = 0, right = n - 1。
- 当 left <= right：
比较中间元素 nums[mid] 和目标值 target 。
	1. 如果 target = nums[mid]，返回 mid。
	2. 如果 target < nums[mid]，则在左侧继续搜索 right = mid - 1。
	3. 如果 target > nums[mid]，则在右侧继续搜索 left = mid + 1。


```go
func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}
```
复杂度分析

- 时间复杂度：O(logN)。
- 空间复杂度：O(1)。



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


[70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)


### 方法一：滚动数组 (斐波那契数列)

![截屏2021-04-09 11.50.02.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpdcsbx511j31940gu756.jpg)

思路和算法

我们用 f(x) 表示爬到第 x 级台阶的方案数，考虑最后一步可能跨了一级台阶，也可能跨了两级台阶，所以我们可以列出如下式子：

f(x) = f(x - 1) + f(x - 2)

f(x) 只和 f(x−1) 与 f(x−2) 有关，所以我们可以用「滚动数组思想」把空间复杂度优化成 O(1)。


```go
func climbStairs(n int) int {
	p, q, r := 0, 0, 1
	for i := 1; i <= n; i++ {
		p = q
		q = r
		r = p + q
	}
	return r
}
```
复杂度分析

- 时间复杂度：循环执行 n 次，每次花费常数的时间代价，故渐进时间复杂度为 O(n)。
- 空间复杂度：这里只用了常数个变量作为辅助空间，故渐进空间复杂度为 O(1)。



### 方法二：动态规划

解题思路 
- 简单的 DP，经典的爬楼梯问题。一个楼梯可以由 n-1 和 n-2 的楼梯爬上来。
- 这一题求解的值就是斐波那契数列。

```go
func climbStairs(n int) int {
	dp := make([]int, n+1)
	dp[0], dp[1] = 1, 1
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}
```

复杂度分析

- 时间复杂度：循环执行 n 次，每次花费常数的时间代价，故渐进时间复杂度为 O(n)。
- 空间复杂度： O(n)。

### 解题思路 
#### 假设 n = 5，有 5 级楼梯要爬
- 题意说，每次有2种选择：爬1级，或爬2级。
	如果爬1级，则剩下4级要爬。
	如果爬2级，则剩下3级要爬。
- 这拆分出了2个问题：
	爬4级楼梯有几种方式？
	爬3级楼梯有几种方式？
- 于是，爬 5 级楼梯的方式数 = 爬 4 级楼梯的方式数 + 爬 3 级楼梯的方式数。
#### 画出递归树
- 用「剩下要爬的楼梯数」描述一个节点。
![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpdsdqld5ij317207674g.jpg)

- 子问题又会面临 2 个选择，不断分支，直到位于递归树底部的 base case：

	楼梯数为 0 时，只有 1 种选择：什么都不做。
	楼梯数为 1 时，只有1种选择：爬1级。

![2.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpdsewfog7j30y60ctaam.jpg)

- 当递归遍历到 base case，解是已知的，开始返回，如下图，子问题的结果不断向上返回，得到父问题的解。

![4.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpdsfh45y0j314h0hkdhr.jpg)

- 调用栈的深度是楼梯数 n，空间复杂度是O(n)O(n)，时间复杂度最坏是O(2^n)，所有节点都遍历到。

### 存在重复的子问题
- 如下图，黄色阴影部分，蓝色阴影部分，是相同的子树，即相同的子问题。

![5.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpdsh8p300j30zr0i7jtk.jpg)

- 子问题的计算结果可以存储在哈希表中，或者数组，下次遇到就不用再进入相同的递归。

- 去除重复的计算后的子树如下，时间复杂度降到了O(n)，空间复杂度O(n)。

![6.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpdsiibb52j30sp0fpabc.jpg)

## 动态规划，自底而上思考
- 我们发现，爬 i 层楼梯的方式数 = 爬 i-2 层楼梯的方式数 + 爬 i-1 层楼梯的方式数。
- 我们有两个 base case，结合上面的递推式，就能递推出爬 i 层楼梯的方式数。
- 用一个数组 dp 存放中间子问题的结果。
- dp[i]：爬 i 级楼梯的方式数。从dp[0]、dp[1]出发，顺序计算，直到算出 dp[i]，就像填表格。

![7.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpdskk5ycoj31c00kx781.jpg)

```go
func climbStairs(n int) int {
	dp := make([]int, n+1)
	dp[0] = 1
	dp[1] = 1
	for i := 2; i < len(dp); i++ {
		dp[i] = dp[i-2] + dp[i-1]
	}
	return dp[n]
}

```

## 压缩空间，优化
dp[i] 只与过去的两项：dp[i-1] 和 dp[i-2] 有关，没有必要存下所有计算过的 dp 项。用两个变量去存这两个过去的状态就好。

```go

func climbStairs(n int) int {
	prev := 1
	cur := 1
	for i := 2; i < n+1; i++ {
		temp := cur
		cur = prev + cur
		prev = temp
	}
	return cur
}

```

### 可以用动态规划的问题都能用递归
- 从子问题入手，解决原问题，分两种做法：自顶向下和自底向上
- 前者对应递归，借助函数调用自己，是程序解决问题的方式，它不会记忆解
- 后者对应动态规划，利用迭代将子问题的解存在数组里，从数组0位开始顺序往后计算
- 递归的缺点在于包含重复的子问题（没有加记忆化的情况下），动态规划的效率更高

### DP也有局限性
- DP 相比于 递归，有时候不太好理解，或者边界情况比较难确定
- 而且必须是一步步邻接的，连续地计算
- 加入了记忆化的递归，就灵活很多，它在递归基础上稍作修改，往往更好理解，也少了局限性，不好用DP时一定能用它
- 比如有时候要求出达到某个结果的路径，递归（DFS）回溯出路径，显然更有优势







[151. 翻转字符串里的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

解题思路 
- 给出一个中间有空格分隔的字符串，要求把这个字符串按照单词的维度前后翻转。
- 依照题意，先把字符串按照空格分隔成每个小单词，然后把单词前后翻转，最后再把每个单词中间添加空格。

```go
func reverseWords(s string) string {
	ss := strings.Fields(s)
	reverse(&ss, 0, len(ss)-1)
	return strings.Join(ss, " ")
}
func reverse(m *[]string, i, j int) {
	for i <= j {
		(*m)[i], (*m)[j] = (*m)[j], (*m)[i]
		i++
		j--
	}
}
```

```go
func reverseWords(s string) string {
	ss := strings.Fields(s) //ss = ["the", "sky", "is", "blue"]
	var reverse func([]string, int, int)
	reverse = func(m []string, i, j int) {
		for i <= j {
			m[i], m[j] = m[j], m[i] // ["blue", "sky", "is", "the"]
			i++
			j--
		}
	}
	reverse(ss, 0, len(ss)-1)
	return strings.Join(ss, " ") //"blue is sky the"
}
```

### func Fields
```go
func Fields(s string) []string
```
返回将字符串按照空白（unicode.IsSpace确定，可以是一到多个连续的空白字符）分割的多个字符串。如果字符串全部是空白或者是空字符串的话，会返回空切片。

```go
Example
fmt.Printf("Fields are: %q", strings.Fields("  foo bar  baz   "))
Output:

Fields are: ["foo" "bar" "baz"]
```

### func Join
```go
func Join(a []string, sep string) string
```
将一系列字符串连接为一个字符串，之间用sep来分隔。

```go
Example
s := []string{"foo", "bar", "baz"}
fmt.Println(strings.Join(s, ", "))
Output:

foo, bar, baz
```




[23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)


```go
func mergeKLists(lists []*ListNode) *ListNode {
	length := len(lists)
	if length < 1 {
		return nil
	}
	if length == 1 {
		return lists[0]
	}
	num := length / 2
	left := mergeKLists(lists[:num])
	right := mergeKLists(lists[num:])
	return mergeTwoLists1(left, right)
}
func mergeTwoLists1(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}
	if l1.Val < l2.Val {
		l1.Next = mergeTwoLists1(l1.Next, l2)
		return l1
	} else {
		l2.Next = mergeTwoLists1(l1, l2.Next)
		return l2
	}
}
```





[104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

### 方法一：深度优先搜索
#### 思路与算法

如果我们知道了左子树和右子树的最大深度 ll 和 rr，那么该二叉树的最大深度即为

max(l,r)+1

而左子树和右子树的最大深度又可以以同样的方式进行计算。因此我们可以用「深度优先搜索」的方法来计算二叉树的最大深度。具体而言，在计算当前二叉树的最大深度时，可以先递归计算出其左子树和右子树的最大深度，然后在 O(1) 时间内计算出当前二叉树的最大深度。递归在访问到空节点时退出。
![](https://assets.leetcode-cn.com/solution-static/104/7.png)
![](https://assets.leetcode-cn.com/solution-static/104/10.png)

```go
func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return max(maxDepth(root.Left), maxDepth(root.Right)) + 1
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

复杂度分析

- 时间复杂度：O(n)，其中 n 为二叉树节点的个数。每个节点在递归中只被遍历一次。
- 空间复杂度：O(height)，其中 height 表示二叉树的高度。递归函数需要栈空间，而栈空间取决于递归的深度，因此空间复杂度等价于二叉树的高度。





[2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

```go
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy := new(ListNode)
	curr, carry := dummy, 0
	for l1 != nil || l2 != nil || carry > 0 {
		curr.Next = new(ListNode)
		curr = curr.Next
		if l1 != nil {
			carry += l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			carry += l2.Val
			l2 = l2.Next
		}
		curr.Val = carry % 10
		carry /= 10
	}
	return dummy.Next
}
```




[110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

### 方法一：自顶向下的递归 (前序遍历)

具体做法类似于二叉树的前序遍历，即对于当前遍历到的节点，首先计算左右子树的高度，如果左右子树的高度差是否不超过 11，再分别递归地遍历左右子节点，并判断左子树和右子树是否平衡。这是一个自顶向下的递归的过程。


```go
func isBalanced(root *TreeNode) bool {
	if root == nil {
		return true
	}
	leftHeight := depth(root.Left)
	rightHeight := depth(root.Right)
	return abs(rightHeight-leftHeight) <= 1 && isBalanced(root.Left) && isBalanced(root.Right)
}
func depth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return max(depth(root.Left), depth(root.Right)) + 1
}
func abs(x int) int {
	if x < 0 {
		return -1 * x
	}
	return x
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

复杂度分析

- 时间复杂度：O(n^2)，其中 nn 是二叉树中的节点个数。
最坏情况下，二叉树是满二叉树，需要遍历二叉树中的所有节点，时间复杂度是 O(n)。
对于节点 p，如果它的高度是 d，则 height(p) 最多会被调用 d 次（即遍历到它的每一个祖先节点时）。对于平均的情况，一棵树的高度 hh 满足 O(h)=O(logn)，因为 d≤h，所以总时间复杂度为 O(nlogn)。对于最坏的情况，二叉树形成链式结构，高度为 O(n)，此时总时间复杂度为 O(n^2)。

- 空间复杂度：O(n)，其中 n 是二叉树中的节点个数。空间复杂度主要取决于递归调用的层数，递归调用的层数不会超过 n。


### 方法二：自底向上的递归 (后序遍历)

方法一由于是自顶向下递归，因此对于同一个节点，函数 height 会被重复调用，导致时间复杂度较高。如果使用自底向上的做法，则对于每个节点，函数 height 只会被调用一次。

自底向上递归的做法类似于后序遍历，对于当前遍历到的节点，先递归地判断其左右子树是否平衡，再判断以当前节点为根的子树是否平衡。如果一棵子树是平衡的，则返回其高度（高度一定是非负整数），否则返回 −1。如果存在一棵子树不平衡，则整个二叉树一定不平衡。

```go
func isBalanced(root *TreeNode) bool {
	return depth(root) >= 0
}
func depth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	leftHeight := depth(root.Left)
	rightHeight := depth(root.Right)
	if leftHeight == -1 || rightHeight == -1 || abs(rightHeight-leftHeight) > 1 {
		return -1
	}
	return max(depth(root.Left), depth(root.Right)) + 1
}
func abs(x int) int {
	if x < 0 {
		return -1 * x
	}
	return x
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```



















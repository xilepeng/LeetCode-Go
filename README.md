

[76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/) next

[129. 求根节点到叶节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)

### 方法一：深度优先搜索
思路与算法

从根节点开始，遍历每个节点，如果遇到叶子节点，则将叶子节点对应的数字加到数字之和。如果当前节点不是叶子节点，则计算其子节点对应的数字，然后对子节点递归遍历。

![](https://assets.leetcode-cn.com/solution-static/129/fig1.png)

![](https://pic.leetcode-cn.com/1603933660-UNWQbT-image.png)

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func sumNumbers(root *TreeNode) int {
	var dfs func(*TreeNode, int) int

	dfs = func(root *TreeNode, prevSum int) int {
		if root == nil {
			return 0
		}
		sum := prevSum*10 + root.Val
		if root.Left == nil && root.Right == nil {
			return sum
		}
		return dfs(root.Left, sum) + dfs(root.Right, sum)
	}

	return dfs(root, 0)
}
```

复杂度分析

- 时间复杂度：O(n)，其中 n 是二叉树的节点个数。对每个节点访问一次。

- 空间复杂度：O(n)，其中 n 是二叉树的节点个数。空间复杂度主要取决于递归调用的栈空间，递归栈的深度等于二叉树的高度，最坏情况下，二叉树的高度等于节点个数，空间复杂度为 O(n)。



[958. 二叉树的完全性检验](https://leetcode-cn.com/problems/check-completeness-of-a-binary-tree/)

### 方法一：广度优先搜索
1. 按 根左右(前序遍历) 顺序依次检查
2. 如果出现空节点，标记end = true
3. 如果后面还有节点，返回false

```go
func isCompleteTree(root *TreeNode) bool {
	q, end := []*TreeNode{root}, false
	for len(q) > 0 {
		node := q[0]
		q = q[1:]
		if node == nil {
			end = true
		} else {
			if end == true {
				return false
			}
			q = append(q, node.Left)
			q = append(q, node.Right)
		}
	}
	return true
}
```


[468. 验证IP地址](https://leetcode-cn.com/problems/validate-ip-address/)



[剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

```go
func exchange(nums []int) []int {
    for i, j := 0, 0; i < len(nums); i++ {
        if nums[i] & 1 == 1 { //nums[i]奇数      nums[j]偶数
            nums[i], nums[j] = nums[j], nums[i] //奇偶交换
            j++
        }
    }
    return nums
}
```

```go
func exchange(nums []int) []int {
    i, j := 0, len(nums)-1
    for i < j {
        for i < j && nums[i] & 1 == 1 { //从左往右、奇数一直向右走
            i++
        }
        for i < j && nums[j] & 1 == 0 { //从右向左、偶数一直向左走
            j--
        }
        nums[i], nums[j] = nums[j], nums[i] //奇偶交换
    }
    return nums
}
```


[322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)


![322. Coin Change and 518. Coin Change 2.png](http://ww1.sinaimg.cn/large/007daNw2ly1gps6k2bgrtj31kg3tub29.jpg)

![截屏2021-04-23 16.55.43.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpts6iwvafj319i0o042z.jpg)

![截屏2021-04-23 13.16.57.png](http://ww1.sinaimg.cn/large/007daNw2ly1gptltwipl8j319q0p2go8.jpg)


#### iterate amount

```go
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	dp[0] = 0 //base case
	for i := 1; i < len(dp); i++ {
		dp[i] = amount + 1
	}
	for i := 1; i <= amount; i++ { //遍历所有状态的所有值
		for _, coin := range coins { //求所有选择的最小值 min(dp[4],dp[3],dp[0])+1
			if i-coin >= 0 {
				dp[i] = min(dp[i], dp[i-coin]+1)
			}
		}
	}
	if amount-dp[amount] < 0 {
		return -1
	}
	return dp[amount]
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```



[518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

![截屏2021-04-23 16.57.11.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpts6y27nhj319a0n8gpb.jpg)

#### iterate coins

```go
func change(amount int, coins []int) int {
	dp := make([]int, amount+1)
	dp[0] = 1
	for _, coin := range coins {
		for i := coin; i <= amount; i++ {
			dp[i] += dp[i-coin]
		}
	}
	return dp[amount]
}
```




[剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)


```go
type CQueue struct {
    inStack, outStack []int
}

func Constructor() CQueue {
    return CQueue{}
}

func (this *CQueue) AppendTail(value int)  {
    this.inStack = append(this.inStack, value)
}

func (this *CQueue) DeleteHead() int {
    if len(this.outStack) == 0 {
        if len(this.inStack) == 0 { return -1}
        for len(this.inStack) > 0 {
            top := this.inStack[len(this.inStack)-1]
            this.inStack = this.inStack[:len(this.inStack)-1]
            this.outStack = append(this.outStack, top)
        }
    }
    top := this.outStack[len(this.outStack)-1]
    this.outStack = this.outStack[:len(this.outStack)-1]
    return top
}

/**
 * Your CQueue object will be instantiated and called as such:
 * obj := Constructor();
 * obj.AppendTail(value);
 * param_2 := obj.DeleteHead();
 */
```



[162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)

### 方法一：二分查找

```go
func findPeakElement(nums []int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := left + (right-left)>>1
		if nums[mid] > nums[mid+1] {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return left
}
```
### 方法二: 线性扫描
```go
func findPeakElement(nums []int) int {
	for i := 0; i < len(nums)-1; i++ {
		if nums[i] > nums[i+1] {
			return i
		}
	}
	return len(nums) - 1
}
```


[179. 最大数](https://leetcode-cn.com/problems/largest-number/)




# 买卖股票的最佳时机

**我们要跳出固有的思维模式，并不是要考虑买还是卖，而是要最大化手里持有的钱。
买股票手里的钱减少，卖股票手里的钱增加，无论什么时刻，我们要保证手里的钱最多。
并且我们这一次买还是卖只跟上一次我们卖还是买的状态有关。**

[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

```go
func maxProfit(prices []int) int {
	buy, sell := math.MinInt64, 0
	for _, p := range prices {
		buy = max(buy, 0-p)
		sell = max(sell, buy+p)
	}
	return sell
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)


```go
func maxProfit(prices []int) int {
	buy, sell := math.MinInt64, 0
	for _, p := range prices {
		buy = max(buy, sell-p)
		sell = max(sell, buy+p)
	}
	return sell
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

这两个问题唯一的不同点在于我们是买一次还是买无穷多次，而代码就只有 0-p 和 sell-p 的区别。
因为如果买无穷多次，就需要上一次卖完的状态。如果只买一次，那么上一个状态一定是0。

[123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)


第三题只允许最多买两次，那么就有四个状态，第一次买，第一次卖，第二次买，第二次卖。
还是那句话，无论什么状态，我们要保证手里的钱最多。

```go
func maxProfit(prices []int) int {
	b1, b2, s1, s2 := math.MinInt64, math.MinInt64, 0, 0
	for _, p := range prices {
		b1 = max(b1, 0-p)
		s1 = max(s1, b1+p)
		b2 = max(b2, s1-p)
		s2 = max(s2, b2+p)
	}
	return s2
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

[188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

```go
func maxProfit(k int, prices []int) int {
	if k >= len(prices)>>1 {
		T_ik0, T_ik1 := 0, math.MinInt64
		for _, price := range prices {
			T_ik0_old := T_ik0
			T_ik0 = max(T_ik0, T_ik1+price)
			T_ik1 = max(T_ik1, T_ik0_old-price)
		}
		return T_ik0
	}
	T_ik0, T_ik1 := make([]int, k+1), make([]int, k+1)
	for i := range T_ik0 {
		T_ik0[i] = 0
		T_ik1[i] = math.MinInt64
	}
	for _, price := range prices {
		for j := k; j > 0; j-- {
			T_ik0[j] = max(T_ik0[j], T_ik1[j]+price)
			T_ik1[j] = max(T_ik1[j], T_ik0[j-1]-price)
		}
	}
	return T_ik0[k]
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

[309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

这道题只是第二题的变形，卖完要隔一天才能买，那么就多记录前一天卖的状态即可。

```go
func maxProfit(prices []int) int {
	buy, sell_pre, sell := math.MinInt64, 0, 0
	for _, p := range prices {
		buy = max(buy, sell_pre-p)
		sell_pre, sell = sell, max(sell, buy+p)
	}
	return sell
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```



[714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

每次买卖需要手续费，那么我们买的时候减掉手续费就行了。
```go
func maxProfit(prices []int, fee int) int {
	buy, sell := math.MinInt64, 0
	for _, p := range prices {
		buy = max(buy, sell-p-fee)
		sell = max(sell, buy+p)
	}
	return sell
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```










[7. 整数反转](https://leetcode-cn.com/problems/reverse-integer/)





[460. LFU 缓存](https://leetcode-cn.com/problems/lfu-cache/)




[补充题6. 手撕堆排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)



[59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/)



[128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)



[剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

```go
func fib(n int) int {
    if n < 2 { 
        return n 
    }
    prev, curr := 0, 1
    for i := 2; i <= n; i++ {
        sum := prev + curr
        prev = curr
        curr = sum % 1000000007
    }
    return curr 
}
```


[509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/)

### 方法一：递归

![截屏2021-04-22 11.34.30.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpsd9a3yaxj317o0o0q7h.jpg)

```go
func fib(n int) int {
	if n == 0 || n == 1 {
		return n
	}
	return fib(n-1) + fib(n-2)
}
```

复杂度分析

- 时间复杂度：O(2^n)。
- 空间复杂度：O(h)。

### 方法二：带备忘录递归


![截屏2021-04-23 20.07.36.png](http://ww1.sinaimg.cn/large/007daNw2ly1gptxon19akj319k0nw407.jpg)

闭包写法：
```go
func fib(n int) int {
	memo := make([]int, n+1)//从0开始
	var helper func(int) int

	helper = func(n int) int {
		if n == 0 || n == 1 {
			return n
		}
		if memo[n] != 0 {
			return memo[n]
		}
		memo[n] = helper(n-1) + helper(n-2)
		return memo[n]
	}

	return helper(n)
}
```

```go
func fib(n int) int {
	memo := make([]int, n+1)
	return helper(memo, n)
}
func helper(memo []int, n int) int {
	if n < 2 {
		return n
	}
	if memo[n] != 0 { //剪枝
		return memo[n]
	}
	memo[n] = helper(memo, n-1) + helper(memo, n-2)
	return memo[n]
}
```

复杂度分析

- 时间复杂度：O(n)。
- 空间复杂度：O(n)。

### 方法三：动态规划

![截屏2021-04-22 11.35.07.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpsda7wdjwj30zu0hwmyn.jpg)

```go
func fib(n int) int {
	if n == 0 {
		return 0
	}
	dp := make([]int, n+1)
	dp[0], dp[1] = 0, 1       //base case
	for i := 2; i <= n; i++ { //状态转移
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}
```
复杂度分析

- 时间复杂度：O(n)。
- 空间复杂度：O(n)。

### 方法四：滚动数组

![截屏2021-04-22 11.42.22.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpsdgjxvorj31520kgjsx.jpg)

动态规划空间优化：只存储前2项

```go
func fib(n int) int {
	if n == 0 || n == 1 { //base case
		return n
	} //递推关系
	prev, curr := 0, 1
	for i := 2; i <= n; i++ {
		next := prev + curr
		prev = curr
		curr = next
	}
	return curr
}
```

复杂度分析

- 时间复杂度：O(n)。
- 空间复杂度：O(1)。

[24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)



```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func swapPairs(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	newHead := head.Next
	head.Next = swapPairs(newHead.Next)
	newHead.Next = head
	return newHead
}
```



```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func swapPairs(head *ListNode) *ListNode {
	dummy := &ListNode{0, head}
	temp := dummy
	for temp.Next != nil && temp.Next.Next != nil {
		node1 := temp.Next
		node2 := temp.Next.Next
		temp.Next = node2
		node1.Next = node2.Next
		node2.Next = node1
		temp = node1
	}
	return dummy.Next
}
```

[32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)



[498. 对角线遍历](https://leetcode-cn.com/problems/diagonal-traverse/)





[328. 奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/)

![](https://pic.leetcode-cn.com/1605227711-BsDKjR-image.png)

思路
- odd 指针扫描奇数结点，even 指针扫描偶数结点

	- 奇数结点逐个改 next，连成奇链
	- 偶数结点逐个改 next，连成偶链
- 循环体内，做 4 件事：

	- 当前奇数结点 ——next——> 下一个奇数结点
	- odd 指针推进 ——————> 下一个奇数结点
	- 当前偶数结点 ——next——> 下一个偶数结点
	- even 指针推进 ——————> 下一个偶数结点

- 扫描结束时，奇链偶链就分开了，此时 odd 指向奇链的尾结点
- 奇链的尾结点 ——next——> 偶链的头结点（循环前保存），就连接了奇偶链

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func oddEvenList(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	evenHead := head.Next
	odd, even := head, evenHead
	for even != nil && even.Next != nil {
		odd.Next = even.Next
		odd = odd.Next
		even.Next = odd.Next
		even = even.Next
	}
	odd.Next = evenHead
	return head
}
```


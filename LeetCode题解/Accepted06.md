


1. [补充题5. 手撕归并排序 912. 排序数组](#补充题5-手撕归并排序-912-排序数组)
2. [460. LFU 缓存](#460-lfu-缓存)
3. [剑指 Offer 09. 用两个栈实现队列](#剑指-offer-09-用两个栈实现队列)
	1. [买卖股票的最佳时机](#买卖股票的最佳时机)
4. [121. 买卖股票的最佳时机](#121-买卖股票的最佳时机)
5. [122. 买卖股票的最佳时机 II](#122-买卖股票的最佳时机-ii)
6. [123. 买卖股票的最佳时机 III](#123-买卖股票的最佳时机-iii)
7. [188. 买卖股票的最佳时机 IV](#188-买卖股票的最佳时机-iv)
8. [309. 最佳买卖股票时机含冷冻期](#309-最佳买卖股票时机含冷冻期)
9. [714. 买卖股票的最佳时机含手续费](#714-买卖股票的最佳时机含手续费)
10. [补充题1. 排序奇升偶降链表](#补充题1-排序奇升偶降链表)
11. [145. 二叉树的后序遍历](#145-二叉树的后序遍历)
12. [198. 打家劫舍](#198-打家劫舍)
	1. [方法一：动态规划](#方法一动态规划)
13. [剑指 Offer 51. 数组中的逆序对](#剑指-offer-51-数组中的逆序对)
14. [138. 复制带随机指针的链表](#138-复制带随机指针的链表)
15. [695. 岛屿的最大面积](#695-岛屿的最大面积)
16. [394. 字符串解码](#394-字符串解码)
17. [209. 长度最小的子数组](#209-长度最小的子数组)
	1. [方法一：滑动窗口](#方法一滑动窗口)
18. [322. 零钱兑换 补充](#322-零钱兑换-补充)
		1. [iterate amount](#iterate-amount)
19. [518. 零钱兑换 II](#518-零钱兑换-ii)
		1. [iterate coins](#iterate-coins)
20. [剑指 Offer 40. 最小的k个数](#剑指-offer-40-最小的k个数)
	1. [方法一：快速选择](#方法一快速选择)
	2. [小根堆](#小根堆)
21. [328. 奇偶链表](#328-奇偶链表)
22. [125. 验证回文串](#125-验证回文串)
23. [189. 旋转数组](#189-旋转数组)
24. [384. 打乱数组](#384-打乱数组)
25. [225. 用队列实现栈](#225-用队列实现栈)


<!-- [补充题5. 手撕归并排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

[460. LFU 缓存](https://leetcode-cn.com/problems/lfu-cache/)

[剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

[补充题1. 排序奇升偶降链表](https://mp.weixin.qq.com/s/377FfqvpY8NwMInhpoDgsw)

[145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)



[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

[剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

[138. 复制带随机指针的链表](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)

[695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

[394. 字符串解码](https://leetcode-cn.com/problems/decode-string/)

[209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)

[322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/) 补充

[518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

[剑指 Offer 40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

[328. 奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/) 

[125. 验证回文串](https://leetcode-cn.com/problems/valid-palindrome/)

[189. 旋转数组](https://leetcode-cn.com/problems/rotate-array/)

[384. 打乱数组](https://leetcode-cn.com/problems/shuffle-an-array/)

[225. 用队列实现栈](https://leetcode-cn.com/problems/implement-stack-using-queues/) -->



------



## [补充题5. 手撕归并排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

```go
func sortArray(nums []int) []int {
	merge_sort(nums, 0, len(nums)-1)
	return nums
}
func merge_sort(A []int, start, end int) {
	if start < end {
		mid := start + (end-start)>>1
		merge_sort(A, start, mid)
		merge_sort(A, mid+1, end)
		merge(A, start, mid, end)
	}
}
func merge(A []int, start, mid, end int) {
	Arr := make([]int, end-start+1)
	p, q, k := start, mid+1, 0
	for i := start; i <= end; i++ {
		if p > mid {
			Arr[k] = A[q]
			q++
		} else if q > end {
			Arr[k] = A[p]
			p++
		} else if A[p] < A[q] {
			Arr[k] = A[p]
			p++
		} else {
			Arr[k] = A[q]
			q++
		}
		k++
	}
	for p := 0; p < k; p++ {
		A[start] = Arr[p]
		start++
	}
}
```

## [460. LFU 缓存](https://leetcode-cn.com/problems/lfu-cache/)

```go
type LFUCache struct {
	cache               map[int]*Node       // 存储缓存的内容
	freq                map[int]*DoubleList // 存储每个频次对应的双向链表
	ncap, size, minFreq int                 // minFreq存储当前最小频次
}

func Constructor(capacity int) LFUCache {
	return LFUCache{
		cache: make(map[int]*Node),
		freq:  make(map[int]*DoubleList),
		ncap:  capacity,
	}
}

func (this *LFUCache) Get(key int) int {
	if node, ok := this.cache[key]; ok {
		this.IncFreq(node)
		return node.val
	}
	return -1
}

func (this *LFUCache) Put(key, value int) {
	if this.ncap == 0 {
		return
	}
	if node, ok := this.cache[key]; ok {
		node.val = value
		this.IncFreq(node)
	} else {
		if this.size >= this.ncap { // 缓存已满，需要进行删除操作
			// 通过 minFreq 拿到 freq_table[minFreq] 链表的末尾节点
			node := this.freq[this.minFreq].RemoveLast()
			delete(this.cache, node.key)
			this.size--
		} // 与 get 操作基本一致，除了需要更新缓存的值
		x := &Node{key: key, val: value, freq: 1}
		this.cache[key] = x
		if this.freq[1] == nil {
			this.freq[1] = CreateDL()
		}
		this.freq[1].AddFirst(x)
		this.minFreq = 1
		this.size++
	}
}

func (this *LFUCache) IncFreq(node *Node) {
	// 从原freq对应的链表里移除, 并更新 minFreq
	_freq := node.freq
	this.freq[_freq].Remove(node)
	// 如果当前链表为空，我们需要在哈希表中删除，且更新 minFreq
	if this.minFreq == _freq && this.freq[_freq].IsEmpty() {
		this.minFreq++
		delete(this.freq, _freq)
	}
	node.freq++ // 插入到 freq + 1 对应的链表中
	if this.freq[node.freq] == nil {
		this.freq[node.freq] = CreateDL()
	}
	this.freq[node.freq].AddFirst(node)
}

type DoubleList struct {
	head, tail *Node
}

type Node struct {
	prev, next     *Node
	key, val, freq int
}

func CreateDL() *DoubleList {
	head, tail := &Node{}, &Node{}
	head.next, tail.prev = tail, head
	return &DoubleList{
		head: head,
		tail: tail,
	}
}

func (this *DoubleList) AddFirst(node *Node) {
	node.next = this.head.next
	node.prev = this.head

	this.head.next.prev = node
	this.head.next = node
}

func (this *DoubleList) Remove(node *Node) {
	node.prev.next = node.next
	node.next.prev = node.prev

	node.next = nil
	node.prev = nil
}

func (this *DoubleList) RemoveLast() *Node {
	if this.IsEmpty() {
		return nil
	}

	last := this.tail.prev
	this.Remove(last)

	return last
}

func (this *DoubleList) IsEmpty() bool {
	return this.head.next == this.tail
}

```






## [剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)


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




###  买卖股票的最佳时机

**我们要跳出固有的思维模式，并不是要考虑买还是卖，而是要最大化手里持有的钱。
买股票手里的钱减少，卖股票手里的钱增加，无论什么时刻，我们要保证手里的钱最多。
并且我们这一次买还是卖只跟上一次我们卖还是买的状态有关。**

## [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

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

## [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)


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

## [123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)


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

## [188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

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

## [309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

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



## [714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

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












## [补充题1. 排序奇升偶降链表](https://mp.weixin.qq.com/s/377FfqvpY8NwMInhpoDgsw)

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func sortOddEvenList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	odd, even := oddEven(head)
	even = reverse(even)
	return merge(odd, even)
}
func oddEven(head *ListNode) (*ListNode, *ListNode) {
	evenHead := head.Next
	odd, even := head, evenHead
	for even != nil && even.Next != nil {
		odd.Next = even.Next
		odd = odd.Next
		even.Next = odd.Next
		even = even.Next
	}
	return odd, even
}
func reverse(head *ListNode) *ListNode {
	var prev *ListNode
	curr := head
	for curr != nil {
		next := curr.Next
		curr.Next = prev
		prev = curr
		curr = next
	}
	return prev
}
func merge(l1, l2 *ListNode) *ListNode {
	dummy := new(ListNode)
	prev := dummy
	for l1 != nil && l2 != nil {
		if l1.Val < l2.Val {
			prev.Next = l1
			l1 = l1.Next
		} else {
			prev.Next = l2
			l2 = l2.Next
		}
		prev = prev.Next
	}
	if l1 != nil {
		prev.Next = l1
	} else {
		prev.Next = l2
	}
	return dummy.Next
}
```







## [145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)



```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func postorderTraversal(root *TreeNode) []int {
	res := []int{}
	var postorder func(*TreeNode)

	postorder = func(root *TreeNode) {
		if root != nil {
			postorder(root.Left)
			postorder(root.Right)
			res = append(res, root.Val)
		}
	}

	postorder(root)
	return res
}
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func postorderTraversal(root *TreeNode) []int {
	res, stack := []int{}, []*TreeNode{}
	p := root
	for len(stack) > 0 || p != nil {
		if p != nil {
			stack = append(stack, p)
			res = append(append([]int{}, p.Val), res...) //反转先序遍历
			p = p.Right                                  //反转先序遍历
		} else {
			node := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			p = node.Left //反转先序遍历
		}
	}
	return res
}
```








## [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

### 方法一：动态规划

解题思路：


状态定义：

设动态规划列表 dp ，dp[i] 代表前 i 个房子在满足条件下的能偷窃到的最高金额。
转移方程：

设： 有 n 个房子，前 n 间能偷窃到的最高金额是 dp[n] ，前 n−1 间能偷窃到的最高金额是 dp[n−1] ，此时向这些房子后加一间房，此房间价值为 num ；

加一间房间后： 由于不能抢相邻的房子，意味着抢第 n+1 间就不能抢第 n 间；那么前 n+1 间房能偷取到的最高金额 dp[n+1] 一定是以下两种情况的 较大值 ：

不抢第 n+1 个房间，因此等于前 n 个房子的最高金额，即 dp[n+1] = dp[n] ；
抢第 n+1 个房间，此时不能抢第 n 个房间；因此等于前 n−1 个房子的最高金额加上当前房间价值，即 dp[n+1]=dp[n−1]+num ；
细心的我们发现： 难道在前 n 间的最高金额 dp[n] 情况下，第 n 间一定被偷了吗？假设没有被偷，那 n+1 间的最大值应该也可能是 dp[n+1] = dp[n] + num 吧？其实这种假设的情况可以被省略，这是因为：

假设第 n 间没有被偷，那么此时 dp[n] = dp[n-1] ，此时 dp[n+1] = dp[n] + num = dp[n-1] + num ，即两种情况可以 合并为一种情况 考虑；
假设第 n 间被偷，那么此时 dp[n+1] = dp[n] + num 不可取 ，因为偷了第 nn 间就不能偷第 n+1 间。
最终的转移方程： dp[n+1] = max(dp[n],dp[n-1]+num)
初始状态：

前 0 间房子的最大偷窃价值为 0 ，即 dp[0] = 0。
返回值：

返回 dp 列表最后一个元素值，即所有房间的最大偷窃价值。
简化空间复杂度：

我们发现 dp[n] 只与 dp[n−1] 和 dp[n−2] 有关系，因此我们可以设两个变量 cur和 pre 交替记录，将空间复杂度降到 O(1) 。

复杂度分析：
- 时间复杂度 O(N) ： 遍历 nums 需要线性时间；
- 空间复杂度 O(1) ： cur和 pre 使用常数大小的额外空间。


```go
func rob(nums []int) int {
	cur, pre := 0, 0
	for _, num := range nums {
		cur, pre = max(pre+num, cur), cur
	}
	return cur
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

复杂度分析

- 时间复杂度：O(n)，其中 n 是数组长度。只需要对数组遍历一次。

- 空间复杂度：O(1)。使用滚动数组，可以只存储前两间房屋的最高总金额，而不需要存储整个数组的结果，因此空间复杂度是 O(1)。







## [剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

```go
func reversePairs(nums []int) int {
	return merge_sort(nums, 0, len(nums)-1)
}

func merge_sort(A []int, start, end int) int {
	if start >= end {
		return 0
	}
	mid := start + (end-start)>>1
	left := merge_sort(A, start, mid)
	right := merge_sort(A, mid+1, end)
	cross := merge(A, start, mid, end)
	return left + right + cross
}
func merge(A []int, start, mid, end int) int {
	Arr := make([]int, end-start+1)
	p, q, k, count := start, mid+1, 0, 0
	for i := start; i <= end; i++ {
		if p > mid {
			Arr[k] = A[q]
			q++
		} else if q > end {
			Arr[k] = A[p]
			p++
		} else if A[p] <= A[q] {
			Arr[k] = A[p]
			p++
		} else {
			count += mid - p + 1
			Arr[k] = A[q]
			q++
		}
		k++
	}
	copy(A[start:end+1], Arr)
	return count
}
```




## [138. 复制带随机指针的链表](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)

![](https://pic.leetcode-cn.com/1789e6dd9bbe41223cab82b2e0a7615cd1a8ed16a3c992462d4e1eaec3b82fb1-image.png)

```go
/**
 * Definition for a Node.
 * type Node struct {
 *     Val int
 *     Next *Node
 *     Random *Node
 * }
 */

func copyRandomList(head *Node) *Node {
	if head == nil {
		return nil
	}
	// 1. 复制每个结点,并放在原结点后
	curr := head
	for curr != nil {
		newNode := &Node{
			Val:    curr.Val,
			Next:   curr.Next,
			Random: curr.Random,
		}
		curr.Next = newNode
		curr = newNode.Next
	}
	// 2.使复制的节点指向正确的随机数。
	curr = head
	for curr != nil {
		curr = curr.Next
		if curr.Random != nil {
			curr.Random = curr.Random.Next
		}
		curr = curr.Next
	}
	// 3. 提取复制的节点并恢复原始列表。
	curr = head
	newHead := head.Next
	for curr.Next != nil {
		temp := curr.Next
		curr.Next = temp.Next //跳过复制节点
		curr = temp
	}
	return newHead
}
```

## [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

```go
func maxAreaOfIsland(grid [][]int) int {
	max_area := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == 1 {
				max_area = max(max_area, dfs(grid, i, j))
			}
		}
	}
	return max_area
}
func dfs(grid [][]int, i, j int) int {
	if i < 0 || j < 0 || i >= len(grid) || j >= len(grid[0]) || grid[i][j] == 0 {
		return 0
	}
	area := 1
	grid[i][j] = 0
	area += dfs(grid, i+1, j)
	area += dfs(grid, i-1, j)
	area += dfs(grid, i, j+1)
	area += dfs(grid, i, j-1)
	return area
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

## [394. 字符串解码](https://leetcode-cn.com/problems/decode-string/)

```go
func decodeString(s string) string {
	numStack := []int{}      // 存倍数的栈
	strStack := []string{}   // 存待拼接的str的栈
	num := 0                 // 倍数的“搬运工”
	res := ""                // 字符串的“搬运工”
	for _, char := range s { // 逐字符扫描
		if char >= '0' && char <= '9' { // 遇到数字
			n, _ := strconv.Atoi(string(char))
			num = num*10 + n // 算出倍数
		} else if char == '[' { // 遇到 [
			strStack = append(strStack, res) // res串入栈
			res = ""                         // 入栈后清零
			numStack = append(numStack, num) // 倍数num进入栈等待
			num = 0                          // 入栈后清零
		} else if char == ']' { // 遇到 ]，两个栈的栈顶出栈
			count := numStack[len(numStack)-1] // 获取拷贝次数
			numStack = numStack[:len(numStack)-1]
			str := strStack[len(strStack)-1]
			strStack = strStack[:len(strStack)-1]
			res = string(str) + strings.Repeat(res, count) // 构建子串
		} else {
			res += string(char) // 遇到字母，追加给res串
		}
	}
	return res
}
```

## [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)


### 方法一：滑动窗口

```go
func minSubArrayLen(target int, nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	res, sum := math.MaxInt64, 0
	start, end := 0, 0
	for end < n {
		sum += nums[end]
		for sum >= target {
			res = min(res, end-start+1)
			sum -= nums[start]
			start++
		}
		end++
	}
	if res == math.MaxInt64 {
		return 0
	}
	return res
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```




复杂度分析

- 时间复杂度：O(n)，其中 n 是数组的长度。指针 start 和 end 最多各移动 n 次。

- 空间复杂度：O(1)。





## [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/) 补充


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
	for i := 1; i <= amount; i++ { //遍历所有(面额)状态的所有值
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



## [518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

![截屏2021-04-23 16.57.11.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpts6y27nhj319a0n8gpb.jpg)

#### iterate coins


```go
func change(amount int, coins []int) int {
	dp := make([]int, amount+1)
	dp[0] = 1
	for _, coin := range coins { //遍历所有硬币
		for i := coin; i <= amount; i++ {
			dp[i] += dp[i-coin]
		}
	}
	return dp[amount]
}
```










## [剑指 Offer 40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

### 方法一：快速选择

```go
func getLeastNumbers(arr []int, k int) []int {
    rand.Seed(time.Now().Unix())
    quickSelect(arr, 0, len(arr)-1, k)
    return arr[:k]
}

func quickSelect(A []int, start, end, k int) {
    if start < end {
        piv_pos := randomPartition(A, start, end)
        if piv_pos == k {
            return
        } else if piv_pos < k {
            quickSelect(A, piv_pos+1, end, k)
        } else {
            quickSelect(A, start, piv_pos-1, k)
        }
    }
}

func partition(A []int, start, end int) int {
    piv, i := A[start], start+1
    for j := start+1; j <= end; j++ {
        if A[j] < piv {
            A[i], A[j] = A[j], A[i]
            i++
        }
    }
    A[start], A[i-1] = A[i-1], A[start]
    return i-1
}

func randomPartition(A []int, start, end int) int {
    random := start + rand.Int()%(end-start+1)
    A[start], A[random] = A[random], A[start]
    return partition(A, start, end)
}

```

复杂度分析

- 时间复杂度：期望为 O(n) ，由于证明过程很繁琐，所以不再这里展开讲。具体证明可以参考《算法导论》第 9 章第 2 小节。

最坏情况下的时间复杂度为 O(n^2)。情况最差时，每次的划分点都是最大值或最小值，一共需要划分 n−1 次，而一次划分需要线性的时间复杂度，所以最坏情况下时间复杂度为 O(n^2)。

- 空间复杂度：期望为 O(logn)，递归调用的期望深度为 O(logn)，每层需要的空间为 O(1)，只有常数个变量。

最坏情况下的空间复杂度为 O(n)。最坏情况下需要划分 n 次，即 randomized_selected 函数递归调用最深 n−1 层，而每层由于需要 O(1) 的空间，所以一共需要 O(n) 的空间复杂度。



### 小根堆

有错

```go
func getLeastNumbers(arr []int, k int) []int {
    if k == 0 {
        return []int{}
    }
    heap_sort(arr, k)
    return arr[len(arr)-k:]
}

func heap_sort(A []int, k int) {
    heap_size := len(A)
    build_minheap(A, heap_size)
    for i := heap_size-1; i >= k-1; i-- {
        A[0], A[i] = A[i], A[0]
        heap_size--
        min_heapify(A, 0, heap_size)
    }
}

func build_minheap(A []int, heap_size int) {
    for i := heap_size>>1; i >= 0; i-- {
        min_heapify(A, i, heap_size)
    }
}

func min_heapify(A []int, i, heap_size int) {
    lson, rson, least := i<<1+1, i<<1+2, i
    for lson < heap_size && A[lson] < A[least] {
        least = lson
    }
    for rson < heap_size && A[rson] < A[least] {
        least = rson
    }
    if least != i {
        A[i], A[least] = A[least], A[i]
        min_heapify(A, least, heap_size)
    }
}
```


## [328. 奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/) 

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


## [125. 验证回文串](https://leetcode-cn.com/problems/valid-palindrome/)

```go
func isPalindrome(s string) bool {
	s = strings.ToLower(s)
	left, right := 0, len(s)-1
	for left < right {
		if !isValid(s[left]) {
			left++
		} else if !isValid(s[right]) {
			right--
		} else {
			if s[left] != s[right] {
				return false
			}
			left++
			right--
		}
	}
	return true
}
func isValid(c byte) bool {
	if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') {
		return true
	}
	return false
}
```


```go
func isPalindrome(s string) bool {
	left, right := 0, len(s)-1
	for left < right {
		if !unicode.IsLetter(rune(s[left])) && !unicode.IsNumber(rune(s[left])) {
			left++
		} else if !unicode.IsLetter(rune(s[right])) && !unicode.IsNumber(rune(s[right])) {
			right--
		} else {
			if unicode.ToLower(rune(s[left])) != unicode.ToLower(rune(s[right])) {
				return false
			}
			left++
			right--
		}
	}
	return true
}
```

## [189. 旋转数组](https://leetcode-cn.com/problems/rotate-array/)

```go
func rotate(nums []int, k int) {
	k %= len(nums)
	reverse(nums)
	reverse(nums[:k])
	reverse(nums[k:])
}
func reverse(nums []int) {
	n := len(nums)
	for i := 0; i < n/2; i++ {
		nums[i], nums[n-1-i] = nums[n-1-i], nums[i]
	}
}
```



## [384. 打乱数组](https://leetcode-cn.com/problems/shuffle-an-array/)



```go

type Solution struct {
	A []int
}

func Constructor(nums []int) Solution {
	// rand.Seed(time.Now().Unix())
	return Solution{nums}
}

/** Resets the array to its original configuration and return it. */
func (this *Solution) Reset() []int {
	return this.A
}

/** Returns a random shuffling of the array. */
func (this *Solution) Shuffle() []int {
	temp := make([]int, len(this.A))
	n := len(temp)
	copy(temp, this.A)
	for i := n - 1; i >= 0; i-- {
		random := rand.Intn(i + 1)
		temp[i], temp[random] = temp[random], temp[i]
	}
	return temp
}

/**
 * Your Solution object will be instantiated and called as such:
 * obj := Constructor(nums);
 * param_1 := obj.Reset();
 * param_2 := obj.Shuffle();
 */
```



```go
func (*Rand) Int
func (r *Rand) Int() int
返回一个非负的伪随机int值。


func (*Rand) Intn
func (r *Rand) Intn(n int) int
返回一个取值范围在[0,n)的伪随机int值，如果n<=0会panic。
```

## [225. 用队列实现栈](https://leetcode-cn.com/problems/implement-stack-using-queues/)

```go
type MyStack struct {
	queue []int
}

/** Initialize your data structure here. */
func Constructor() (s MyStack) {
	return
}

/** Push element x onto stack. */
func (s *MyStack) Push(x int) {
	n := len(s.queue)
	s.queue = append(s.queue, x)
	for ; n > 0; n-- {
		s.queue = append(s.queue, s.queue[0])
		s.queue = s.queue[1:]
	}
}

/** Removes the element on top of the stack and returns that element. */
func (s *MyStack) Pop() int {
	v := s.queue[0]
	s.queue = s.queue[1:]
	return v
}

/** Get the top element. */
func (s *MyStack) Top() int {
	return s.queue[0]
}

/** Returns whether the stack is empty. */
func (s *MyStack) Empty() bool {
	return len(s.queue) == 0
}

/**
 * Your MyStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * param_2 := obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.Empty();
 */
```

```go
type MyStack struct {
	queue1, queue2 []int
}

/** Initialize your data structure here. */
func Constructor() (s MyStack) {
	return
}

/** Push element x onto stack. */
func (s *MyStack) Push(x int) {
	s.queue2 = append(s.queue2, x)
	for len(s.queue1) > 0 {
		s.queue2 = append(s.queue2, s.queue1[0])
		s.queue1 = s.queue1[1:]
	}
	s.queue1, s.queue2 = s.queue2, s.queue1
}

/** Removes the element on top of the stack and returns that element. */
func (s *MyStack) Pop() int {
	v := s.queue1[0]
	s.queue1 = s.queue1[1:]
	return v
}

/** Get the top element. */
func (s *MyStack) Top() int {
	return s.queue1[0]
}

/** Returns whether the stack is empty. */
func (s *MyStack) Empty() bool {
	return len(s.queue1) == 0
}

/**
 * Your MyStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * param_2 := obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.Empty();
 */
```


[460. LFU 缓存](https://leetcode-cn.com/problems/lfu-cache/)

[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

[补充题5. 手撕归并排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

[补充题1. 排序奇升偶降链表](https://mp.weixin.qq.com/s/377FfqvpY8NwMInhpoDgsw)

[145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)

[22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

[剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)












[328. 奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/) 





------

[460. LFU 缓存](https://leetcode-cn.com/problems/lfu-cache/)

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










###  买卖股票的最佳时机

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









[补充题5. 手撕归并排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

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




[补充题1. 排序奇升偶降链表](https://mp.weixin.qq.com/s/377FfqvpY8NwMInhpoDgsw)

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







[145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)

![Binary Tree Traversal Iteration Implementation.png](http://ww1.sinaimg.cn/large/007daNw2ly1gqnx5to70yj30pd2bie81.jpg)


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
	if root == nil {
		return []int{}
	}
	stack, res := []*TreeNode{root}, []int{}
	for len(stack) != 0 {
		curr := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append(append([]int{}, curr.Val), res...)
		if curr.Left != nil {
			stack = append(stack, curr.Left)
		}
		if curr.Right != nil {
			stack = append(stack, curr.Right)
		}

	}
	return res
}
```



[22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)







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





[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

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



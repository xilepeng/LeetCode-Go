
[剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

[41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)

[剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

[322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

[876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/) 补充

[24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

[468. 验证IP地址](https://leetcode-cn.com/problems/validate-ip-address/)

[剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

[43. 字符串相乘](https://leetcode-cn.com/problems/multiply-strings/)

[补充题6. 手撕堆排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

[179. 最大数](https://leetcode-cn.com/problems/largest-number/)

[128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

[460. LFU 缓存](https://leetcode-cn.com/problems/lfu-cache/)

[498. 对角线遍历](https://leetcode-cn.com/problems/diagonal-traverse/)

[227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/)

[32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)

[补充题1. 排序奇升偶降链表](https://mp.weixin.qq.com/s/377FfqvpY8NwMInhpoDgsw)

[59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/)

[162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)

[剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)




------


[剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

```go
/*
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}
*/
var head, prev *TreeNode

func treeToDoublyList(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	dfs(root)
	head.Left, prev.Right = prev, head
	return head
}
func dfs(curr *TreeNode) {
	if curr == nil {
		return
	}
	dfs(curr.Left)
	if prev == nil {
		head = curr
	} else {
		prev.Right, curr.Left = curr, prev
	}
	prev = curr
	dfs(curr.Right)
}

```

[41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/) 

```go

```




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




[468. 验证IP地址](https://leetcode-cn.com/problems/validate-ip-address/)





[876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/) 补充

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func middleNode(head *ListNode) *ListNode {
    slow, fast := head, head 
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    return slow
}
```

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


[468. 验证IP地址](https://leetcode-cn.com/problems/validate-ip-address/) next





[剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func kthLargest(root *TreeNode, k int) (res int) {
    var dfs func(*TreeNode)

    dfs = func(root *TreeNode) {
        if root == nil {
            return
        }
        dfs(root.Right)
        k--
        if k == 0 { res = root.Val}
        dfs(root.Left) 
    }
    
    dfs(root)
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
func kthLargest(root *TreeNode, k int) int {
    res := []int{}
    var dfs func(*TreeNode)

    dfs = func(root *TreeNode) {
        if root == nil {
            return
        }
        dfs(root.Right)
        res = append(res, root.Val)
        dfs(root.Left)
    }

    dfs(root)
    return res[k-1]
}
```


[43. 字符串相乘](https://leetcode-cn.com/problems/multiply-strings/)

```go

```

[补充题6. 手撕堆排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)


思路和算法

堆排序的思想就是先将待排序的序列建成大根堆，使得每个父节点的元素大于等于它的子节点。此时整个序列最大值即为堆顶元素，我们将其与末尾元素交换，使末尾元素为最大值，然后再调整堆顶元素使得剩下的 n−1 个元素仍为大根堆，再重复执行以上操作我们即能得到一个有序的序列。


```go
func sortArray(nums []int) []int {
	heapSort(nums)
	return nums
}
func heapSort(a []int) {
	heapSize := len(a)
	buildMaxHeap(a, heapSize)
	for i := heapSize - 1; i >= 0; i-- {
		a[0], a[i] = a[i], a[0] //堆顶元素和堆底元素交换
		heapSize--              //把剩余待排序元素整理成堆
		maxHeapify(a, 0, heapSize)
	}
}
func buildMaxHeap(a []int, heapSize int) { // O(n)
	for i := heapSize / 2; i >= 0; i-- { // 从后往前调整所有非叶子节点
		maxHeapify(a, i, heapSize)
	}
}
func maxHeapify(a []int, i, heapSize int) { // O(nlogn)
	l, r, largest := i*2+1, i*2+2, i
	if l < heapSize && a[largest] < a[l] { //左儿子存在且大于a[largest]
		largest = l
	}
	if r < heapSize && a[largest] < a[r] { //右儿子存在且大于a[largest]
		largest = r
	}
	if largest != i {
		a[largest], a[i] = a[i], a[largest] //堆顶调整为最大值
		maxHeapify(a, largest, heapSize)    //递归处理
	}
}
```

复杂度分析

- 时间复杂度：O(nlogn)。初始化建堆的时间复杂度为 O(n)，建完堆以后需要进行 n−1 次调整，一次调整（即 maxHeapify） 的时间复杂度为 O(logn)，那么 n−1 次调整即需要 O(nlogn) 的时间复杂度。因此，总时间复杂度为 O(n+nlogn)=O(nlogn)。

- 空间复杂度：O(1)。只需要常数的空间存放若干变量。






[179. 最大数](https://leetcode-cn.com/problems/largest-number/)

[128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

[460. LFU 缓存](https://leetcode-cn.com/problems/lfu-cache/)

[498. 对角线遍历](https://leetcode-cn.com/problems/diagonal-traverse/)

[227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/)

[32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)



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


[59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/)



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
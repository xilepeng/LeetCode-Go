CodeTop排名

1. [剑指 Offer 22. 链表中倒数第k个节点](#剑指-offer-22-链表中倒数第k个节点)
2. [剑指 Offer 09. 用两个栈实现队列](#剑指-offer-09-用两个栈实现队列)
3. [剑指 Offer 36. 二叉搜索树与双向链表](#剑指-offer-36-二叉搜索树与双向链表)
4. [剑指 Offer 54. 二叉搜索树的第k大节点](#剑指-offer-54-二叉搜索树的第k大节点)
5. [剑指 Offer 06. 从尾到头打印链表](#剑指-offer-06-从尾到头打印链表)
6. [剑指 Offer 10- I. 斐波那契数列](#剑指-offer-10--i-斐波那契数列)
7. [剑指 Offer 10- II. 青蛙跳台阶问题](#剑指-offer-10--ii-青蛙跳台阶问题)
8. [剑指 Offer 30. 包含min函数的栈](#剑指-offer-30-包含min函数的栈)
9. [剑指 Offer 04. 二维数组中的查找](#剑指-offer-04-二维数组中的查找)
10. [剑指 Offer 42. 连续子数组的最大和](#剑指-offer-42-连续子数组的最大和)
11. [剑指 Offer 51. 数组中的逆序对](#剑指-offer-51-数组中的逆序对)
12. [剑指 Offer 40. 最小的k个数](#剑指-offer-40-最小的k个数)
13. [剑指 Offer 18. 删除链表的节点](#剑指-offer-18-删除链表的节点)
14. [剑指 Offer 52. 两个链表的第一个公共节点](#剑指-offer-52-两个链表的第一个公共节点)
15. [剑指 Offer 25. 合并两个排序的链表](#剑指-offer-25-合并两个排序的链表)
16. [剑指 Offer 10- I. 斐波那契数列](#剑指-offer-10--i-斐波那契数列-1)
17. [剑指 Offer 10- II. 青蛙跳台阶问题](#剑指-offer-10--ii-青蛙跳台阶问题-1)
18. [剑指 Offer 48. 最长不含重复字符的子字符串](#剑指-offer-48-最长不含重复字符的子字符串)
19. [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](#剑指-offer-21-调整数组顺序使奇数位于偶数前面)
20. [剑指 Offer 62. 圆圈中最后剩下的数字](#剑指-offer-62-圆圈中最后剩下的数字)
21. [剑指 Offer 26. 树的子结构](#剑指-offer-26-树的子结构)
22. [剑指 Offer 27. 二叉树的镜像](#剑指-offer-27-二叉树的镜像)
23. [剑指 Offer 29. 顺时针打印矩阵](#剑指-offer-29-顺时针打印矩阵)
24. [剑指 Offer 52. 两个链表的第一个公共节点](#剑指-offer-52-两个链表的第一个公共节点-1)
25. [剑指 Offer 61. 扑克牌中的顺子](#剑指-offer-61-扑克牌中的顺子)
26. [剑指 Offer 39. 数组中出现次数超过一半的数字](#剑指-offer-39-数组中出现次数超过一半的数字)
27. [剑指 Offer 45. 把数组排成最小的数](#剑指-offer-45-把数组排成最小的数)
28. [剑指 Offer 34. 二叉树中和为某一值的路径](#剑指-offer-34-二叉树中和为某一值的路径)
29. [剑指 Offer 53 - I. 在排序数组中查找数字 I](#剑指-offer-53---i-在排序数组中查找数字-i)
30. [剑指 Offer 48. 最长不含重复字符的子字符串](#剑指-offer-48-最长不含重复字符的子字符串-1)
31. [剑指 Offer 11. 旋转数组的最小数字](#剑指-offer-11-旋转数组的最小数字)
32. [剑指 Offer 33. 二叉搜索树的后序遍历序列](#剑指-offer-33-二叉搜索树的后序遍历序列)
33. [剑指 Offer 03. 数组中重复的数字](#剑指-offer-03-数组中重复的数字)
34. [剑指 Offer 32 - III. 从上到下打印二叉树 III](#剑指-offer-32---iii-从上到下打印二叉树-iii)
35. [剑指 Offer 07. 重建二叉树 📝](#剑指-offer-07-重建二叉树-)
36. [剑指 Offer 35. 复杂链表的复制 📝](#剑指-offer-35-复杂链表的复制-)
37. [剑指 Offer 24. 反转链表](#剑指-offer-24-反转链表)



## [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode.cn/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)


```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func getKthFromEnd(head *ListNode, k int) *ListNode {
    slow, fast := head, head
    for fast != nil {
        if k > 0 {
            fast = fast.Next // fast 指针先走 k 步
            k--
        } else {
            slow = slow.Next
            fast = fast.Next
        }
    }
    return slow
}

func getKthFromEnd1(head *ListNode, k int) *ListNode {
    slow, fast := head, head
    for i := 0; fast != nil; i++ {
        if i >= k {
            slow = slow.Next
        }
        fast = fast.Next // fast 指针先走 k 步
    }
    return slow
}
```


## [剑指 Offer 09. 用两个栈实现队列](https://leetcode.cn/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

```go
type CQueue struct {
    inStack,outStack []int 
}

func Constructor() CQueue {
    return CQueue{
    }
}

func (this *CQueue) AppendTail(value int)  {
    this.inStack = append(this.inStack, value)
}

func (this *CQueue) DeleteHead() int {
    if len(this.outStack) == 0 {
        if len(this.inStack) == 0 {
            return -1
        }
        this.in2out()
    }
    head := this.outStack[len(this.outStack)-1]
    this.outStack = this.outStack[:len(this.outStack)-1]
    return head
}

func (this *CQueue) in2out(){
    for len(this.inStack) > 0 {
        this.outStack = append(this.outStack, this.inStack[len(this.inStack)-1])
        this.inStack = this.inStack[:len(this.inStack)-1]
    }
}

/**
 * Your CQueue object will be instantiated and called as such:
 * obj := Constructor();
 * obj.AppendTail(value);
 * param_2 := obj.DeleteHead();
 */
```


## [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode.cn/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/description/)

```go
package main

import "fmt"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

var head, prev *TreeNode // head 头 prev 尾

// 二叉搜索树转双向循环链表
func treeToDoublyList(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	dfs(root)
	// 首尾相连使其成为双向循环链表  head->1<->2<->3<->4<->5<-pre    head<->prev
	prev.Right = head // 尾指向头
	head.Left = prev  // 头指向尾
	return head
}

// 中序遍历: 左根右
func dfs(curr *TreeNode) {
	if curr == nil {
		return
	}
	dfs(curr.Left)   // 递归左子树
	if prev == nil { // 记录头节点
		head = curr
	} else { // prev != nil
		prev.Right = curr // 修改单向 -> 节点引用为双向 <->（双向循环链表）
		curr.Left = prev
	}
	prev = curr     // 滚动扫描下一个节点
	dfs(curr.Right) // 递归右子树
}

func main() {
	// 构造二叉搜索树
	root := &TreeNode{4, nil, nil}
	node1 := &TreeNode{2, nil, nil}
	node2 := &TreeNode{5, nil, nil}
	node3 := &TreeNode{1, nil, nil}
	node4 := &TreeNode{3, nil, nil}
	root.Left = node1
	root.Right = node2
	node1.Left = node3
	node1.Right = node4

	head := treeToDoublyList(root)
	tail := head.Left

	fmt.Println("从头开始遍历:")
	for i := 0; i <= 9; i++ {
		fmt.Printf(" <-> %d", head.Val)
		head = head.Right
	}
	fmt.Println("\n从尾开始遍历:")
	for i := 0; i <= 9; i++ {
		fmt.Printf(" <-> %d", tail.Val)
		tail = tail.Left
	}

}

//从头开始遍历:
//<-> 1 <-> 2 <-> 3 <-> 4 <-> 5 <-> 1 <-> 2 <-> 3 <-> 4 <-> 5
//从尾开始遍历:
//<-> 5 <-> 4 <-> 3 <-> 2 <-> 1 <-> 5 <-> 4 <-> 3 <-> 2 <-> 1

```




## [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)



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




## [剑指 Offer 06. 从尾到头打印链表](https://leetcode.cn/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reversePrint(head *ListNode) []int {
    res := []int{}
    head = reverseList(head)
    for ; head != nil; head = head.Next {
        res = append(res, head.Val)
    }
    return res
}

func reverseList(head *ListNode) *ListNode{
    dummy := &ListNode{Next:head}
    curr := head
    for curr != nil && curr.Next != nil {
        next := curr.Next
        curr.Next = next.Next
        next.Next = dummy.Next
        dummy.Next = next
    }
    return dummy.Next
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
func reversePrint(head *ListNode) (res []int) {
    if head == nil {
        return 
    }
    res = reversePrint(head.Next)
    res = append(res, head.Val)
    return 
}
func reversePrint1(head *ListNode) (res []int) {
    stack := []int{}
    for ; head != nil; head=head.Next {
        stack = append(stack, head.Val)
    }
    for i := len(stack)-1; i >= 0; i-- {
        res = append(res, stack[i])
    }
    return
}
```




## [剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)



```go
func fib(n int) int {
    prev, curr := 0, 1 
    for ; n > 0; n-- {
        sum := (prev + curr) % 1000000007
        prev = curr
        curr = sum
    }
    return prev
}

func fib1(n int) int {
    if n == 0 || n == 1 {
        return n 
    }
    prev, curr := 0, 1
    for i := 2; i <= n; i++ {
        next := (prev + curr) % 1000000007
        prev = curr
        curr = next
    }
    return curr
}
```



## [剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

```go
func numWays(n int) int {
	prev, curr := 1, 1
	for ; n > 0; n-- {
		sum := (prev + curr) % 1000000007
		prev = curr
		curr = sum
	}
	return prev
}

func numWays1(n int) int {
    if n == 1 || n == 2 {
        return n 
    }
	prev, curr := 1, 1
	for i := 2; i <= n; i++{
		sum := (prev + curr) % 1000000007
		prev = curr
		curr = sum
	}
	return curr
} 

func numWays2(n int) int {
	prev, curr := 1, 1
	for i := 2; i <= n; i++{
		sum := (prev + curr) % 1000000007
		prev = curr
		curr = sum
	}
	return curr
} 
```



## [剑指 Offer 30. 包含min函数的栈](https://leetcode.cn/problems/bao-han-minhan-shu-de-zhan-lcof/)


```go
type MinStack struct {
	stack    []int
	minStack []int
}

/** initialize your data structure here. */
func Constructor() MinStack {
	return MinStack{
		stack:    []int{},
		minStack: []int{math.MaxInt64}, // 单调栈：单调递减，top 存储最小值
	}
}

func (this *MinStack) Push(x int) {
	this.stack = append(this.stack, x)
	top := this.minStack[len(this.minStack)-1]
	this.minStack = append(this.minStack, min(top, x))
}

func (this *MinStack) Pop() {
	this.stack = this.stack[:len(this.stack)-1]
	this.minStack = this.minStack[:len(this.minStack)-1]
}

func (this *MinStack) Top() int {
	return this.stack[len(this.stack)-1]
}

func (this *MinStack) Min() int {
	return this.minStack[len(this.minStack)-1]
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

/**
 * Your MinStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(val);
 * obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.Min();
 */
```

```go
type MinStack struct {
	stack    []int
	minStack []int // 单调栈：单调递减，top 存储最小值
}

/** initialize your data structure here. */
func Constructor() MinStack {
	return MinStack{
		nil,
		nil,
	}
}

func (this *MinStack) Push(x int) {
	this.stack = append(this.stack, x)
	if len(this.minStack) == 0 || this.minStack[len(this.minStack)-1] >= x {
		this.minStack = append(this.minStack, x)
	}
}

func (this *MinStack) Pop() {
	if this.minStack[len(this.minStack)-1] == this.stack[len(this.stack)-1] {
		this.minStack = this.minStack[:len(this.minStack)-1]
	}
	this.stack = this.stack[:len(this.stack)-1]
}

func (this *MinStack) Top() int {
	return this.stack[len(this.stack)-1]
}

func (this *MinStack) Min() int {
	return this.minStack[len(this.minStack)-1]
}

/**
 * Your MinStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.Min();
 */
```


## [剑指 Offer 04. 二维数组中的查找](https://leetcode.cn/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/description/)


```go
func findNumberIn2DArray(matrix [][]int, target int) bool {
    if len(matrix) == 0 {
        return false
    }
    x, y := 0, len(matrix[0])-1
    for x < len(matrix) && y >= 0 {
        if matrix[x][y] == target {
            return true
        }
        if target < matrix[x][y] {
            y--
        } else {
            x++
        }
    }
    return false
}
```





## [剑指 Offer 42. 连续子数组的最大和](https://leetcode.cn/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)


```go
func maxSubArray(nums []int) int {
	max, preSum := math.MinInt32, 0 // max = nums[0] OK
	for _, x := range nums {
		if preSum < 0 {
			preSum = 0
		}
		preSum += x
		if max < preSum {
			max = preSum
		}
	}
	return max
}
```


```go
func maxSubArray(nums []int) int {
	pre, maxSum := 0, nums[0]
	for _, x := range nums {
        // 若当前指针所指元素之前的和小于0，则丢弃当前元素之前的数列
		pre = max(pre+x, x)       
		maxSum = max(maxSum, pre) // 将当前值与最大值比较，取最大
	}
	return maxSum
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

```go
func maxSubArray(nums []int) int {
	max := nums[0]
	for i := 1; i < len(nums); i++ {
        // 若前一个元素大于0，将其加到当前元素上
		if nums[i-1]+nums[i] > nums[i] { // nums[i-1] > 0  
			nums[i] += nums[i-1]
		}
		if max < nums[i] {
			max = nums[i]
		}
	}
	return max
}
```




## [剑指 Offer 51. 数组中的逆序对](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)


```go
func reversePairs(nums []int) int {
	return mergeSort(nums, 0, len(nums)-1)
}

func mergeSort(arr []int, start, end int) int {
	if start >= end {
		return 0
	}
	mid := start + (end-start)>>1
	left := mergeSort(arr, start, mid)
	right := mergeSort(arr, mid+1, end)
	cross := merge(arr, start, mid, end)
	return left + right + cross
}

func merge(arr []int, start, mid, end int) int {
	tmpArr := make([]int, end-start+1)
	i, j, k, count := start, mid+1, 0, 0
	for p := start; p <= end; p++ {
		if i > mid {
			tmpArr[k] = arr[j]
			j++
		} else if j > end {
			tmpArr[k] = arr[i]
			i++
		} else if arr[i] <= arr[j] {
			tmpArr[k] = arr[i]
			i++
		} else {
			count += mid - i + 1
			tmpArr[k] = arr[j]
			j++
		}
		k++
	}
	copy(arr[start:end+1], tmpArr)
	return count
}
```




## [剑指 Offer 40. 最小的k个数](https://leetcode.cn/problems/zui-xiao-de-kge-shu-lcof/description/)

**1. 结果无序**

```go
func getLeastNumbers(arr []int, k int) []int {
    quickSelect(arr, 0, len(arr)-1, k)
    return arr[:k]
}

func quickSelect(A []int, low, high, k int) {
    if low >= high {
        return
    }
    pos := partition(A, low, high)
    if pos == k {
        return
    } else if pos < k {
        quickSelect(A, pos+1, high, k)
    } else {
        quickSelect(A, low, pos-1, k)
    }
}

func partition(A []int, low, high int) int {
    A[high], A[low+(high-low)>>1] = A[low+(high-low)>>1], A[high] // 优化
    i, j := low, high 
    for i < j {
        for i < j && A[i] <= A[high] {
            i++
        }
        for i < j && A[j] >= A[high] {
            j--
        }
        A[i], A[j] = A[j], A[i]
    }
    A[i], A[high] = A[high], A[i]
    return i 
}

func Partition(A []int, low, high int) int {
    i, x := low, A[high]
    for j := low; j < high; j++ {
        if A[j] < x {
            A[i], A[j] = A[j], A[i]
            i++
        }
    }
    A[i], A[high] = A[high], A[i]
    return i
}
```


**2. 结果有序**

```go
func getLeastNumbers(A []int, k int) (res []int) {
	heapSize := len(A)
	buildHeap(A, heapSize)
	for i := heapSize - 1; i >= 0; i-- {
		if k == 0 { // 优化
			break
		}
		res = append(res, A[0])
		k--
		A[0], A[i] = A[i], A[0]
		heapSize--
		minHeapify(A, 0, heapSize)
	}
	return
}

func buildHeap(A []int, heapSize int) {
	for i := heapSize >> 1; i >= 0; i-- {
		minHeapify(A, i, heapSize)
	}
}

// 小根堆：降序
func minHeapify(A []int, i, heapSize int) {
	for i<<1+1 < heapSize {
		lson, rson, small := i<<1+1, i<<1+2, i
		for lson < heapSize && A[lson] < A[small] {
			small = lson
		}
		for rson < heapSize && A[rson] < A[small] {
			small = rson
		}
		if small != i {
			A[i], A[small] = A[small], A[i]
			i = small
		} else {
			break
		}
	}
}
```


## [剑指 Offer 18. 删除链表的节点](https://leetcode.cn/problems/shan-chu-lian-biao-de-jie-dian-lcof/)


```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func deleteNode(head *ListNode, val int) *ListNode {
	dummy := &ListNode{Next: head}
	prev, curr := dummy, dummy.Next
	for curr != nil{
		if curr.Val == val {
            prev.Next = curr.Next
        } 
        prev = prev.Next
        curr = curr.Next
	}
	return dummy.Next
}
```


## [剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode.cn/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	if headA == nil || headB == nil {
		return nil
	}
	A, B := headA, headB
	for A != B {
		if A != nil {
			A = A.Next
		} else {
			A = headB
		}
		if B != nil {
			B = B.Next
		} else {
			B = headA
		}
	}
	return A
}
```










## [剑指 Offer 25. 合并两个排序的链表](https://leetcode.cn/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)


```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy := &ListNode{}
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
	if l1 == nil {
		prev.Next = l2
	} else {
		prev.Next = l1
	}
	return dummy.Next
}

func mergeTwoLists1(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	} else if l2 == nil {
		return l1
	} else if l1.Val < l2.Val {
		l1.Next = mergeTwoLists(l1.Next, l2)
		return l1
	} else {
		l2.Next = mergeTwoLists(l1, l2.Next)
		return l2
	}
}
```





## [剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)



```go
func fib(n int) int {
    prev, curr := 0, 1 
    for ; n > 0; n-- {
        sum := (prev + curr)%1000000007
        prev = curr
        curr = sum
    }
    return prev
}
```


``` go
func fib(n int) int {
    if n == 0 || n == 1 {
        return n 
    }
    prev, curr := 0, 1
    for i := 2; i <= n; i++ {
        next := (prev + curr)%1000000007
        prev = curr
        curr = next
    }
    return curr
}
```



## [剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

``` go
func numWays(n int) int {
	prev, curr := 1, 1
	for ; n > 0; n-- {
		sum := (prev + curr) % 1000000007
		prev = curr
		curr = sum
	}
	return prev
}


func numWays1(n int) int {
    if n == 1 || n == 2 {
        return n 
    }
	prev, curr := 1, 1
	for i := 2; i <= n; i++{
		sum := (prev + curr) % 1000000007
		prev = curr
		curr = sum
	}
	return curr
} 

func numWays2(n int) int {
	prev, curr := 1, 1
	for i := 2; i <= n; i++{
		sum := (prev + curr) % 1000000007
		prev = curr
		curr = sum
	}
	return curr
} 
```




## [剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode.cn/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof)

```go
func lengthOfLongestSubstring(s string) int {
	longest, n := 0, len(s)
	freq := make(map[byte]int, n) // 哈希集合记录每个字符出现次数
	for i, j := 0, 0; j < n; j++ {
		freq[s[j]]++         // 首次出现存入哈希
		for freq[s[j]] > 1 { // 当前字符与首字符重复
			freq[s[i]]-- // 收缩窗口，跳过重复首字符
			i++ 		 // 向后扫描
			if freq[s[j]] == 1 { // 优化：如果无重复退出循环
				break
			}
		}
		if longest < j-i+1 { // 统计无重复字符的最长子串
			longest = j - i + 1
		}
	}
	return longest
}
```


## [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode.cn/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)




```go
func exchange(nums []int) []int {
    n := len(nums)
    for slow, fast := 0, 0; fast < n; fast++ {
        if nums[fast]&1 == 1 { // 奇数
            nums[slow], nums[fast] = nums[fast], nums[slow] // 奇数交换到前半部分
            slow++
        }
    }
    return nums
}
```


**方法一：两次遍历**
思路

新建一个数组 res 用来保存调整完成的数组。遍历两次 nums，第一次遍历时把所有奇数依次追加到 res 中，第二次遍历时把所有偶数依次追加到 res 中。

```go
func exchange(nums []int) []int {
    res := make([]int, 0, len(nums))
    for _,num := range nums {
        if num%2 == 1 {
            res = append(res, num)
        }
    }
    for _,num := range nums {
        if num%2 == 0 {
            res = append(res, num)
        }
    }
    return res
}
```

**复杂度分析**

- 时间复杂度：O(n)，其中 n 为数组 nums 的长度。需遍历 nums 两次。

- 空间复杂度：O(1)。结果不计入空间复杂度。


**方法二：双指针 + 一次遍历**
```go
func exchange(nums []int) []int {
    n := len(nums)
    res := make([]int, n)
    left, right := 0, n-1
    for _,num := range nums {
        if num%2 == 1 {
            res[left] = num
            left++
        } else {
            res[right] = num
            right--
        }
    }
    return res
}
```
**复杂度分析**

- 时间复杂度：O(n)，其中 n 为数组 nums 的长度。需遍历 nums 1次。

- 空间复杂度：O(1)。结果不计入空间复杂度。


**方法三：原地交换**
```go
func exchange(nums []int) []int {
    left, right := 0, len(nums)-1
    for left < right {
        for left < right && nums[left]%2 == 1 {
            left++
        }
        for left < right && nums[right]%2 == 0 {
            right--
        }
		if left < right {
            nums[left], nums[right] = nums[right], nums[left]	
			//left++    
			//right--
        }
    }
    return nums
}
```


## [剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode.cn/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/description/)

递推关系：(新 + m) % n

```go
func lastRemaining(n int, m int) int {
	if n == 1 {
		return 0
	}
	return (lastRemaining(n-1, m) + m) % n
}
```

```go
func lastRemaining(n int, m int) int {
	return f(n, m)
}

func f(n int, m int) int {
	if n == 1 {
		return 0
	}
	x := f(n-1, m)
	return (x + m) % n
}
```


## [剑指 Offer 26. 树的子结构](https://leetcode.cn/problems/shu-de-zi-jie-gou-lcof/description/)

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func isSubStructure(A *TreeNode, B *TreeNode) bool {
	if A == nil || B == nil { // 约定空树不是任意一个树的子结构
		return false
	}
	if isSame(A, B) {
		return true
	}
	return isSubStructure(A.Left, B) || isSubStructure(A.Right, B)
}

func isSame(A *TreeNode, B *TreeNode) bool {
	if B == nil {
		return true
	}
	if A == nil || A.Val != B.Val {
		return false
	}
	return isSame(A.Left, B.Left) && isSame(A.Right, B.Right)
}
```


## [剑指 Offer 27. 二叉树的镜像](https://leetcode.cn/problems/er-cha-shu-de-jing-xiang-lcof/description/)


![](../已完成/2022-LeetCode题解/images/Howell.png)

**思路**：

- 遍历的过程中反转每个节点的左右孩子就可以达到整体反转的效果；
- 注意：只要把每一个结点的左右孩子都反转一下，就可以达到整体反转的效果；
- 前序、后序、层序都可以，中序不可以，因为中序遍历会把某些节点的孩子反转2次；

**解法一：递归**

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func mirrorTree(root *TreeNode) *TreeNode {
    if root == nil {
        return nil 
    } // 前序遍历：根左右
    root.Left, root.Right = root.Right, root.Left // 反转当前节点的左右孩子节点：交换左右孩子节点
    mirrorTree(root.Left) // 反转左子树
    mirrorTree(root.Right)// 反转右子树
    return root
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
func mirrorTree(root *TreeNode) *TreeNode {
    if root == nil {
        return nil 
    } // 后序遍历：左右根
    mirrorTree(root.Left) // 反转左子树
    mirrorTree(root.Right)// 反转右子树
    root.Left, root.Right = root.Right, root.Left // 反转当前节点的左右孩子节点：交换左右孩子节点
    return root
}
```

**解法二：迭代法**

- 模拟前序遍历

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func mirrorTree(root *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    q := []*TreeNode{root}
    for len(q) > 0 {
        node := q[0]
        q = q[1:]
        node.Left, node.Right = node.Right, node.Left // 前序遍历：根左右
        if node.Left != nil {
            q = append(q, node.Left)
        }
        if node.Right != nil {
            q = append(q, node.Right)
        }
    }
    return root
}
```

## [剑指 Offer 29. 顺时针打印矩阵](https://leetcode.cn/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

```go
func spiralOrder(matrix [][]int) (res []int) {
	if len(matrix) == 0 {
		return
	}
	top, right, bottom, left := 0, len(matrix[0])-1, len(matrix)-1, 0
	for left <= right && top <= bottom { // 一条边从头遍历到底 (包括最后一个元素)
		for i := left; i <= right; i++ { // 上
			res = append(res, matrix[top][i])
		}
		top++                            // 四个边界同时收缩，进入内层
		for i := top; i <= bottom; i++ { // 右
			res = append(res, matrix[i][right])
		}
		right--
		if top > bottom || left > right { // 防止重复遍历
			break
		}
		for i := right; i >= left; i-- { // 下
			res = append(res, matrix[bottom][i])
		}
		bottom--
		for i := bottom; i >= top; i-- { // 左
			res = append(res, matrix[i][left])
		}
		left++
	}
	return
}
```

## [剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode.cn/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	A, B := headA, headB
	for A != B {
		if A != nil {
			A = A.Next
		} else { // 首次遍历到尾，以另一个链表的头为起点从头开始遍历
			A = headB
		}
		if B != nil {
			B = B.Next
		} else {
			B = headA
		}
	}
	return A // 直到链表相交 A == B，退出循环返回
}
```


## [剑指 Offer 61. 扑克牌中的顺子](https://leetcode.cn/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

```go
func isStraight(nums []int) bool {
	m := make(map[int]int, 5)
	min, max := 14, 0
	for _, num := range nums {
		if num == 0 { // 跳过大小王
			continue
		}
		if num < min { // 最小牌
			min = num
		}
		if num > max { // 最大牌
			max = num
		}
		if _, ok := m[num]; ok { // 若有重复，提前返回 false
			return false
		}
		m[num] = num // 添加牌至 Set
	}
	return max-min < 5 // 最大牌 - 最小牌 < 5 则可构成顺子
}
```



```go
func isStraight(nums []int) bool {
	joker := 0
	sort.Ints(nums) // 数组排序
	for i := 0; i < len(nums)-1; i++ {
		if nums[i] == 0 { // 统计大小王数量
			joker++
		} else if nums[i] == nums[i+1] { // 若有重复，提前返回 false
			return false
		}
	}
	return nums[4]-nums[joker] < 5 // 最大牌 - 最小牌 < 5 则可构成顺子
}
```


![参考](https://leetcode.cn/problems/bu-ke-pai-zhong-de-shun-zi-lcof/solutions/212071/mian-shi-ti-61-bu-ke-pai-zhong-de-shun-zi-ji-he-se/)



## [剑指 Offer 39. 数组中出现次数超过一半的数字](https://leetcode.cn/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)


```go
func majorityElement(nums []int) int {
	res, count := -1, 0
	for _, num := range nums {
		if count == 0 { // 如果票数等于0，重新赋值，抵消掉非众数
			res = num
		}
		if res == num { // 如果num和众数res相等,票数自增1
			count++
		} else { // 不相等,票数自减1
			count--
		}
	}
	return res
}
```

```go
func moreThanHalfNum_Solution(nums []int) int {
    major, vote := -1, 0
    for _, x := range nums {
        if vote == 0 {
            major = x 
        }
        if major == x {
            vote++
        } else {
            vote--
        }
    }
    return major
}
```


## [剑指 Offer 45. 把数组排成最小的数](https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

```go

func minNumber(nums []int) string {
    // 快排实现排序，排序后转成string
	res := make([]string, len(nums))
	for i, v := range nums {
		res[i] = strconv.Itoa(v)
	}
	compare := func(str1, str2 string) bool {
		nums1, _ := strconv.Atoi(str1 + str2)
		nums2, _ := strconv.Atoi(str2 + str1)
		if nums1 < nums2 {
			return true
		}
		return false
	}
	var quickSort func([]string, int, int)

	quickSort = func(strArr []string, low, high int) {
		if low > high {
			return
		}
		i, j := low, len(strArr)-1
		pivot := strArr[high]
		for i < j {
			for i < j && compare(strArr[i], pivot) {
				i++
			}
			for i < j && !compare(strArr[j], pivot) {
				j--
			}
			strArr[i], strArr[j] = strArr[j], strArr[i]
		}
		strArr[i], strArr[high] = strArr[high], strArr[i]

		quickSort(strArr, low, i-1)
		quickSort(strArr, i+1, high)
	}

	quickSort(res, 0, len(res)-1)

	ans := ""
	for _, s := range res {
		ans += s
	}
	return ans
}
```

![参考题解](https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/solutions/190476/mian-shi-ti-45-ba-shu-zu-pai-cheng-zui-xiao-de-s-4/)

！[参考视频](https://www.acwing.com/video/182/)


## [剑指 Offer 34. 二叉树中和为某一值的路径](https://leetcode.cn/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func pathSum(root *TreeNode, target int) (res [][]int) {
    path := []int{}
    var preorder func(*TreeNode, int)
    
    preorder = func(node *TreeNode, left int) {
        if node == nil {
            return
        }
        left -= node.Val
        path = append(path, node.Val)
        defer func(){path = path[:len(path)-1]}()
        if left == 0 && node.Left == nil && node.Right == nil {
            res = append(res, append([]int(nil), path...))
            return
        }
        preorder(node.Left, left)
        preorder(node.Right, left)
    }

    preorder(root, target)
    return 
}
```

## [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode.cn/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/description/)

```go
// 二分查找 target 和 target−1 的右边界，将两结果相减并返回即可。
func search(nums []int, target int) int {
	var searchFirstGreaterElement func(int) int // 第一个大于 target的数的下标
	searchFirstGreaterElement = func(target int) int {
		low, high := 0, len(nums)-1
		for low <= high {
			mid := low + ((high - low) >> 1)
			if nums[mid] <= target {
				low = mid + 1
			} else {
				high = mid - 1
			}
		}
		return low
	}
	return searchFirstGreaterElement(target) - searchFirstGreaterElement(target-1)
}
```

![参考](https://leetcode.cn/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/solutions/155893/mian-shi-ti-53-i-zai-pai-xu-shu-zu-zhong-cha-zha-5/)


## [剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode.cn/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

```go
func lengthOfLongestSubstring(s string) int {
	longest, n := 0, len(s)
	freq := make(map[byte]int, n) // 哈希集合记录每个字符出现次数
	for i, j := 0, 0; j < n; j++ {
		freq[s[j]]++         // 首次出现存入哈希
		for freq[s[j]] > 1 { // 当前字符与首字符重复
			freq[s[i]]-- // 收缩窗口，跳过重复首字符
			i++ 		 // 向后扫描
			if freq[s[j]] == 1 { // 优化：如果无重复退出循环
				break
			}
		}
		if longest < j-i+1 { // 统计无重复字符的最长子串
			longest = j - i + 1
		}
	}
	return longest
}
```



## [剑指 Offer 11. 旋转数组的最小数字](https://leetcode.cn/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/description/)

```go
func minArray(nums []int) int {
	low, high := 0, len(nums)-1
	for low < high {
		mid := low + ((high - low) >> 1) // mid = (low + high)/2
		if nums[mid] < nums[high] {      // mid 在右排序区，旋转点在[low, mid]
			high = mid
		} else if nums[mid] > nums[high] { // mid 在左排序区，旋转点在[mid+1, high]
			low = mid + 1
		} else { // 无法判断 mid 在哪个排序数组中
			high--
		}
	}
	return nums[low]
}
```


## [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/description/)


```go
func verifyPostorder(postorder []int) bool {
    var dfs func(int, int) bool

    dfs = func(i, j int)bool {
        if i >= j {   // 数组中元素最多只有一个
            return true
        }
        left := i     // 左子树下标
        for postorder[left] < postorder[j] { // postorder[i:right-1] < 根节点的值 
            left++
        }
        right := left // 右子树下标
        for postorder[right] > postorder[j] { // postorder[right:j-1] > 根节点的值 
            right++
        }
        return right == j && dfs(i, right-1) && dfs(right, j-1) // 递归判断左右子树
    }

    return dfs(0, len(postorder)-1)
}



func verifyPostorder2(postorder []int) bool {
    var dfs func(int, int) bool

    dfs = func(i, j int)bool {
        if i >= j {   // 数组中元素最多只有一个
            return true
        }
        pos := i     // 左子树下标
        for postorder[pos] < postorder[j] {     // postorder[i:pos-1] < 根节点的值 
            pos++
        }
        for right := pos ; right < j; right++ {  // pos 右子树下标
            if postorder[right] < postorder[j] { // postorder[pos:j-1] > 根节点的值 
                return false
            }
        }
        return dfs(i, pos-1) && dfs(pos, j-1) // 递归判断左右子树
    }

    return dfs(0, len(postorder)-1)
}
```





## [剑指 Offer 03. 数组中重复的数字](https://leetcode.cn/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)



**解法一：原地交换**

思路：把每个数放到对应的位置上，即让 nums[i] = i。
从前往后遍历数组中的所有数，假设当前遍历到的数是 nums[i]，那么：

- 如果 nums[i] != i && nums[nums[i]] == nums[i]，则说明 nums[i] 出现了多次，直接返回 nums[i] 即可；
- 如果 nums[nums[i]] != nums[i]，那我们就把 x 交换到正确的位置上，即 swap(nums[nums[i]], nums[i])，每次swap操作都会将一个数放在正确的位置上，
- 交换完之后如果nums[i] != i，则重复进行该操作。由于每次交换都会将一个数放在正确的位置上，所以swap操作最多会进行 n 次，不会发生死循环。
循环结束后，如果没有找到任何重复的数，则返回-1。

```go
func findRepeatNumber(nums []int) int {
	for i := range nums {
		for nums[nums[i]] != nums[i] { // 如果 nums[i] 不在 nums[i] 位置
			nums[nums[i]], nums[i] = nums[i], nums[nums[i]] // 每次swap操作都会将一个数放在正确的位置上
		}
		if nums[i] != i { // 如果出现重复，直接返回
			return nums[i]
		}
	}
	return -1
}
```


**解法二：原地交换**

```go
func findRepeatNumber(nums []int) int {
	for i := 0; i < len(nums); {
		if nums[i] == i { // 此数字已在对应索引位置，无需交换，因此跳过
			i++
			continue
		}
		if nums[nums[i]] == nums[i] { // 重复：索引 nums[i] 处和索引 i 处的元素值都为 nums[i]
			return nums[i]            // 即找到一组重复值，返回此值 nums[i]
		}
		nums[nums[i]], nums[i] = nums[i], nums[nums[i]] // 将此数字交换至对应索引位置
	}
	return -1
}
```

[参考](https://leetcode.cn/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/solutions/96623/mian-shi-ti-03-shu-zu-zhong-zhong-fu-de-shu-zi-yua/)






## [剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode.cn/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func levelOrder(root *TreeNode) (res [][]int) {
    var bfs func(*TreeNode, int)

    bfs = func(node *TreeNode, level int) {
        if node == nil {
            return 
        }
        if level == len(res) {
            res = append(res, []int{})
        }
        if level%2 == 0 { // 第一层按照从左到右的顺序打印
            res[level] = append(res[level],node.Val )
        } else {          // 第二层按照从右到左的顺序打印
            res[level] = append([]int{node.Val}, res[level]...)
        }
        bfs(node.Left, level+1)
        bfs(node.Right, level+1)
    }
    bfs(root, 0)
    return
}
```


## [剑指 Offer 07. 重建二叉树](https://leetcode.cn/problems/zhong-jian-er-cha-shu-lcof/) 📝


```go

```


## [剑指 Offer 35. 复杂链表的复制](https://leetcode.cn/problems/fu-za-lian-biao-de-fu-zhi-lcof/) 📝

```go

```





## [剑指 Offer 24. 反转链表](https://leetcode.cn/problems/fan-zhuan-lian-biao-lcof/)

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode{
    dummy := &ListNode{Next:head}
    curr := head
    for curr != nil && curr.Next != nil {
        next := curr.Next
        curr.Next = next.Next
        next.Next = dummy.Next
        dummy.Next = next
    }
    return dummy.Next
}

func reverseList1(head *ListNode) *ListNode {
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
```





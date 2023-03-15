

1. [剑指 Offer 22. 链表中倒数第k个节点](#剑指-offer-22-链表中倒数第k个节点)
2. [剑指 Offer 09. 用两个栈实现队列](#剑指-offer-09-用两个栈实现队列)
3. [剑指 Offer 36. 二叉搜索树与双向链表](#剑指-offer-36-二叉搜索树与双向链表)
4. [剑指 Offer 54. 二叉搜索树的第k大节点](#剑指-offer-54-二叉搜索树的第k大节点)
5. [剑指 Offer 03. 数组中重复的数字](#剑指-offer-03-数组中重复的数字)
6. [剑指 Offer 06. 从尾到头打印链表](#剑指-offer-06-从尾到头打印链表)
7. [剑指 Offer 24. 反转链表](#剑指-offer-24-反转链表)
8. [剑指 Offer 10- I. 斐波那契数列](#剑指-offer-10--i-斐波那契数列)
9. [剑指 Offer 10- II. 青蛙跳台阶问题](#剑指-offer-10--ii-青蛙跳台阶问题)
10. [剑指 Offer 30. 包含min函数的栈](#剑指-offer-30-包含min函数的栈)
11. [剑指 Offer 04. 二维数组中的查找](#剑指-offer-04-二维数组中的查找)
12. [剑指 Offer 33. 二叉搜索树的后序遍历序列](#剑指-offer-33-二叉搜索树的后序遍历序列)



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
            fast = fast.Next
            k--
        } else {
            slow, fast = slow.Next, fast.Next
        }
    }
    return slow
}
func getKthFromEnd0(head *ListNode, k int) *ListNode {
    slow, fast := head, head
    for i := 0; fast != nil; i++ {
        if i >= k {
            slow = slow.Next
        }
        fast = fast.Next
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



## [剑指 Offer 30. 包含min函数的栈](https://leetcode.cn/problems/bao-han-minhan-shu-de-zhan-lcof/)


```go

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
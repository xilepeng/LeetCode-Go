

1. [剑指 Offer 03. 数组中重复的数字](#剑指-offer-03-数组中重复的数字)
2. [剑指 Offer 09. 用两个栈实现队列](#剑指-offer-09-用两个栈实现队列)
3. [剑指 Offer 06. 从尾到头打印链表](#剑指-offer-06-从尾到头打印链表)
4. [剑指 Offer 24. 反转链表](#剑指-offer-24-反转链表)



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
	i, n := 0, len(nums)
	for i < n {
		if nums[i] == i { // 此数字已在对应索引位置，无需交换，因此跳过
			i++
			continue
		}
		if nums[nums[i]] == nums[i] { // 索引 nums[i] 处和索引 i 处的元素值都为 nums[i]
			return nums[i]            // 即找到一组重复值，返回此值 nums[i]
		}
		nums[nums[i]], nums[i] = nums[i], nums[nums[i]] // 将此数字交换至对应索引位置
	}
	return -1
}
```

[参考](https://leetcode.cn/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/solutions/96623/mian-shi-ti-03-shu-zu-zhong-zhong-fu-de-shu-zi-yua/)



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
[206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/) 

[25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)


------
[206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/) 


### 方法一：双指针（迭代）

![截屏2021-04-13 14.14.02.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpi39qgmnoj311w0o6n2k.jpg)


思路：将当前节点的 next 指针改为指向前一个节点。

``` go
func reverseList(head *ListNode) *ListNode {
	var prev *ListNode
	curr := head
	for curr != nil {
		next := curr.Next //1.存储下一个节点
		curr.Next = prev  //2.反转
		prev = curr       //3.移动双指针
		curr = next
	}
	return prev
}
```

复杂度分析

- 时间复杂度：O(n)，其中 n 是链表的长度。需要遍历链表一次。

- 空间复杂度：O(1)。



### 方法二：头插法

![截屏2021-04-21 12.11.01.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpr8ocoatoj316s0o844c.jpg)

![截屏2021-04-21 12.11.12.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpr8otljdfj315c0oy42w.jpg)


``` go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
	dummy, curr := &ListNode{Next: head}, head
	for curr.Next != nil {
		next := curr.Next
		curr.Next = next.Next
		next.Next = dummy.Next
		dummy.Next = next
	}
	return dummy.Next
}
```


### 方法三：递归

![截屏2021-04-13 14.08.51.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpi34y26nwj312u0lejug.jpg)

递归版本稍微复杂一些，其关键在于反向工作。假设链表的其余部分已经被反转，现在应该如何反转它前面的部分？

假设链表为：

n1 → … → nk−1 → nk → nk+1 → … → nm → ∅

若从节点 nk+1 到 nm 已经被反转，而我们正处于 nk。
n1 → … → nk−1 → nk → nk+1 ← … ← nm
​	
我们希望 nk+1 的下一个节点指向 nk。

所以，nk.next.next = nk 。
需要注意的是 n1 的下一个节点必须指向 ∅。如果忽略了这一点，链表中可能会产生环。



#### 思路

首先我们先考虑 reverseList 函数能做什么，它可以翻转一个链表，并返回新链表的头节点，也就是原链表的尾节点。
所以我们可以先递归处理 reverseList(head->next)，这样我们可以将以 head->next 为头节点的链表翻转，并得到原链表的尾节 tail，此时 head->next 是新链表的尾节点，我们令它的 next 指针指向 head，并将 head->next 指向空即可将整个链表翻转，且新链表的头节点是tail。

- 空间复杂度分析：总共递归 n 层，系统栈的空间复杂度是 O(n)，所以总共需要额外 O(n) 的空间。
- 时间复杂度分析：链表中每个节点只被遍历一次，所以时间复杂度是 O(n)。


``` go
func reverseList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil { //只有一个节点或没有节点
		return head
	}
	newHead := reverseList(head.Next) //head.Next后 已反转
	head.Next.Next = head             //反转head
	head.Next = nil
	return newHead
}
```

复杂度分析
- 时间复杂度：O(n)，其中 n 是链表的长度。需要对链表的每个节点进行反转操作。

- 空间复杂度：O(n)，其中 n 是链表的长度。空间复杂度主要取决于递归调用的栈空间，最多为 n 层。





[25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

![截屏2021-04-21 11.31.37.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpr7jmcf05j315g0pen05.jpg)

``` go
func reverseKGroup(head *ListNode, k int) *ListNode {
	dummy := &ListNode{Next: head}
	prev := dummy
	for head != nil {
		for i := 0; i < k-1 && head != nil; i++ {
			head = head.Next
		}
		if head == nil {
			break
		}
		curr := prev.Next
		next := head.Next
		head.Next = nil
		prev.Next = reverse(curr)
		curr.Next = next
		head = next
		prev = curr
	}
	return dummy.Next
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
```

复杂度分析

- 时间复杂度：O(n)，其中 n 为链表的长度。head 指针会在 O(⌊k/n⌋) 个节点上停留，每次停留需要进行一次 O(k) 的翻转操作。

- 空间复杂度：O(1)，我们只需要建立常数个变量。






[237. Delete Node in a Linked List](https://leetcode.com/problems/delete-node-in-a-linked-list/)

``` go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func deleteNode(node *ListNode) {
	*node = *node.Next
}
```
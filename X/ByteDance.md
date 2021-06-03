

[25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

![截屏2021-04-21 11.31.37.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpr7jmcf05j315g0pen05.jpg)


```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
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
		first := prev.Next         //待翻转链表头
		next := head.Next          //存储下一个待翻转链表头
		head.Next = nil            //断开
		prev.Next = reverse(first) //翻转
		first.Next = next          //连接
		prev = first
		head = next
	}
	return dummy.Next
}
func reverse(head *ListNode) *ListNode {
	var prev *ListNode
	curr := head
	for curr != nil {
		next := curr.Next
		curr.Next = prev //翻转
		prev = curr
		curr = next
	}
	return prev
}
```


/*
 * @lc app=leetcode.cn id=206 lang=golang
 *
 * [206] 反转链表
 */

// @lc code=start
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode {
	var prev *ListNode //prev -> nil
	curr := head
	for curr != nil {
		next := curr.Next //1.存储下一个节点
		curr.Next = prev  //2.当前节点的next指针指向前一个节点
		prev = curr       //3.移动双指针
		curr = next
	}
	return prev
}

// @lc code=end


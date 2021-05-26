







[237. Delete Node in a Linked List](https://leetcode.com/problems/delete-node-in-a-linked-list/)

```go
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
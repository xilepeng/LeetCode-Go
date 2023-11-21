package main

import "fmt"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

var pre, head *TreeNode

// 二叉搜索树转双向循环链表
func treeToDoublyList(curr *TreeNode) *TreeNode {
	if curr == nil {
		return nil
	}
	dfs(curr)
	head.Left = pre
	pre.Right = head
	return head
}

func dfs(curr *TreeNode) {
	if curr == nil {
		return
	}
	dfs(curr.Left)
	if pre == nil {
		head = curr
	} else {
		pre.Right = curr
		curr.Left = pre
	}
	pre = curr
	dfs(curr.Right)
}

// 输出双向循环链表
func outPut(root *TreeNode) {
	for i := 0; i < 10; i++ {
		fmt.Print("<->", root.Val)
		root = root.Right
	}
}

func main() {
	// 构造二叉搜索树
	root := &TreeNode{4, nil, nil}
	node1 := &TreeNode{1, nil, nil}
	node2 := &TreeNode{2, nil, nil}
	node3 := &TreeNode{3, nil, nil}
	node5 := &TreeNode{5, nil, nil}
	root.Left = node2
	root.Right = node5
	node2.Left = node1
	node2.Right = node3

	root = treeToDoublyList(root) // 二叉搜索树转双向循环链表

	outPut(root) // 输出双向循环链表
	// <->1<->2<->3<->4<->5<->1<->2<->3<->4<->5
}

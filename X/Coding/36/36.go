package main

import "fmt"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

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

func main() {
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
	//从头开始遍历
	for i := 0; i <= 9; i++ {
		fmt.Printf("%d\t", head.Val)
		head = head.Right
	}
	//从尾开始遍历
	for i := 0; i <= 9; i++ {
		fmt.Printf("%d\t", tail.Val)
		tail = tail.Left
	}

}

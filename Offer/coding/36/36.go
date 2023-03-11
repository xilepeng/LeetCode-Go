package main

import "fmt"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

var head, prev *TreeNode // head 头 prev 尾

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
func dfs(curr *TreeNode) { // 中序遍历: 左根右
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

// 输出双向循环链表
func outPut(root *TreeNode) {
	for i := 0; i < 10; i++ {
		if root != nil {
			fmt.Print("<->", root.Val)
		}
		root = root.Right
	}
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

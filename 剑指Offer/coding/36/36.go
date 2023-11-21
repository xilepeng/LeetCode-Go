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
